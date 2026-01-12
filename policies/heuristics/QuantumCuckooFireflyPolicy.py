from ..base_policy import BasePolicy
import numpy as np
import random
import math


class QICFFPolicy(BasePolicy):
    def __init__(self, env, config):
        self.env = env

        # Poids du score
        self.alpha = 0.1
        self.beta = 0.1
        self.gamma = 0.6

        # Firefly
        self.population_size = 100
        self.attractiveness = 1.0
        self.gamma_firefly = 1.0
        self.quantum_noise = 0.05

        # Cuckoo Search
        self.pa = 0.25 # probabilité d'abandon de nids
        self.levy_scale = 1.0
        self.iterations = 10

    # ==============================
    # Score
    # ==============================

    def _score_node(self, node, task):
        node_obj = self.env.scenario.get_node(node)

        # --- 1. Paramètres de base et Sécurisation ---
        ddl = getattr(task, 'DDL', getattr(task, 'deadline', getattr(task, 'ddl', 50.0)))
        cpu_speed = max(node_obj.free_cpu_freq, 1e-6)
        task_size = getattr(task, 'task_size', getattr(task, 'TaskSize', 100))
        cpb = getattr(task, 'cycles_per_bit', getattr(task, 'CyclesPerBit', 50.0))

        # --- 2. Calcul des composants du score ---

        # A. Latence et Énergie
        transmission_time = task_size / max(task.trans_bit_rate, 1e-6)
        computation_time = (task_size * cpb) / cpu_speed
        wait_time = len(node_obj.task_buffer.task_ids) * computation_time
        total_latency = transmission_time + computation_time + wait_time
        energy_consumed = computation_time * node_obj.exe_energy_coef

        # B. Débit (Throughput) : Nombre de tâches traitées par seconde
        # Plus le débit est grand, plus le nœud est puissant pour cette tâche
        tasks_per_second = cpu_speed / (task_size * cpb)

        # C. Pénalité de tâche active
        # On vérifie si le nœud travaille déjà sur quelque chose
        active_penalty = 0.2 if len(node_obj.active_task_ids) > 0 else 0.0

        # --- 3. Normalisation (Toutes les valeurs vers [0, 1]) ---

        # Latence : saturée par la DDL
        s_latency = 1 - math.exp(-total_latency / (ddl * 0.7))

        # Énergie : saturée par une ref de 0.1J
        s_energy = 1 - math.exp(-energy_consumed / 0.3)

        # Débit : On veut qu'un débit ÉLEVÉ donne un score BAS (meilleur).
        s_throughput = math.exp(-tasks_per_second / 0.5)

        # --- 4. Calcul Final ---
        # On ajoute la pénalité d'active task directement au score
        score = (
                self.alpha * s_latency +
                self.beta * s_energy +
                self.gamma / s_throughput +
                active_penalty
        )

        # Pénalité de dépassement DDL (Incontournable)
        if total_latency > ddl:
            score += 0.4

        return min(score, 1.0)

    # ==============================
    # Voisin
    # ==============================
    def _get_neighbor(self, node_idx, all_nodes):
        # On cherche un nœud qui est "proche" dans la liste des nœuds disponibles
        # Cela simule un mouvement local
        current_pos = all_nodes.index(node_idx)
        # On bouge de +/- 1 ou 2 positions dans la liste
        step = random.choice([-2, -1, 1, 2])
        new_pos = (current_pos + step) % len(all_nodes)
        return all_nodes[new_pos]


    # ==============================
    # Lévy Flight
    # ==============================
    def _levy_flight(self, current_node_idx, all_nodes):
        """
        Calcule un saut de Lévy et retourne l'indice du nouveau nœud cible.
        """
        beta = 1.5

        # Calcul du paramètre sigma pour la distribution normale
        sigma_num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        sigma_den = math.gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))
        sigma = (sigma_num / sigma_den)**(1 / beta)

        # Génération de deux distributions normales (u et v)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)

        # Calcul du pas de Lévy (Lévy step)
        step = u / (abs(v)**(1 / beta))

        # Facteur d'échelle pour l'adapter à la taille du réseau
        # On multiplie par levy_scale pour ajuster l'amplitude du saut
        step_size = self.levy_scale * step * (current_node_idx - random.choice(all_nodes))

        # Nouvel indice (on arrondit et on s'assure de rester dans les bornes)
        new_idx = int(abs(current_node_idx + step_size)) % len(all_nodes)

        return all_nodes[new_idx]


    # ==============================
    # Action
    # ==============================

    def act(self, env, task, **kwargs):
        nodes = [
            i for i in range(1, len(env.scenario.node_id2name))
            if env.scenario.get_node(env.scenario.node_id2name[i]).free_cpu_freq > 0
        ]

        if not nodes:
            return 0, None

        # 1. Initialisation
        pop = []
        for _ in range(min(self.population_size, len(nodes))):
            node_idx = random.choice(nodes)
            pop.append({
                'id': node_idx,
                'score': self._score_node(env.scenario.node_id2name[node_idx], task)
            })

        for _ in range(self.iterations):
            pop = sorted(pop, key=lambda x: x['score'])

            # --- PHASE FIREFLY ---
            for i in range(len(pop)):
                for j in range(i):
                    if pop[i]['score'] > pop[j]['score']:
                        r = abs(pop[i]['score'] - pop[j]['score'])
                        attraction = self.attractiveness * math.exp(-self.gamma_firefly * r**2)
                        if random.random() < attraction:
                            pop[i]['id'] = self._get_neighbor(pop[j]['id'], nodes)

            # --- PHASE CUCKOO (Lévy Flight) ---
            for i in range(int(len(pop) * self.pa)):
                idx_to_modify = random.randint(0, len(pop)-1)
                pop[idx_to_modify]['id'] = self._levy_flight(pop[idx_to_modify]['id'], nodes)

            # --- PHASE QUANTUM & UPDATE ---
            for p in pop:
                if random.random() < self.quantum_noise:
                    p['id'] = random.choice(nodes)
                    # Mise à jour du score après mouvements
                p['score'] = self._score_node(env.scenario.node_id2name[p['id']], task)

        best = min(pop, key=lambda x: x['score'])
        return best['id'], None