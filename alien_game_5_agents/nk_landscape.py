import itertools
import random


ATTRIBUTE_NAMES = [
    "alpha", "beta", "gamma", "delta", "epsilon",
    "zeta", "eta", "theta", "iota", "kappa",
]


class NKLandscape:
    def __init__(self, N=10, K=5, seed=None):
        if not (0 <= K <= N - 1):
            raise ValueError(f"K must be between 0 and N-1={N-1}, got K={K}")

        self.N = N
        self.K = K
        self.seed = seed
        self._rng = random.Random(seed)

        self.interaction_map = self._build_interaction_map()
        self.contribution_tables = self._build_contribution_tables()
        self._all_configs_cache = None

    def _build_interaction_map(self):
        interaction_map = {}
        for i in range(self.N):
            others = [j for j in range(self.N) if j != i]
            partners = self._rng.sample(others, self.K)
            interaction_map[i] = partners
        return interaction_map

    def _build_contribution_tables(self):
        tables = {}
        for i in range(self.N):
            num_relevant = self.K + 1
            num_rows = 2 ** num_relevant
            tables[i] = [self._rng.uniform(0, 1) for _ in range(num_rows)]
        return tables

    def get_payoff(self, config):
        if len(config) != self.N:
            raise ValueError(f"Configuration must have length {self.N}")

        contributions = []
        for i in range(self.N):
            relevant_indices = [i] + self.interaction_map[i]
            relevant_states = [config[idx] for idx in relevant_indices]
            row_index = int("".join(str(bit) for bit in relevant_states), 2)
            contributions.append(self.contribution_tables[i][row_index])

        return sum(contributions) / self.N

    def enumerate_all_configs(self):
        if self._all_configs_cache is not None:
            return self._all_configs_cache

        all_configs = []
        for combo in itertools.product([0, 1], repeat=self.N):
            payoff = self.get_payoff(combo)
            all_configs.append((combo, payoff))

        all_configs.sort(key=lambda pair: pair[1])
        self._all_configs_cache = all_configs
        return all_configs

    def get_global_optimum(self):
        all_configs = self.enumerate_all_configs()
        return all_configs[-1]

    def get_lowest_performing_configuration(self):
        all_configs = self.enumerate_all_configs()
        return all_configs[0]

    def get_streetlight_seed(self, percentile_low, percentile_high, seed=None):
        all_configs = self.enumerate_all_configs()
        n = len(all_configs)
        low_idx = int(percentile_low * n)
        high_idx = int(percentile_high * n)
        candidates = all_configs[low_idx:high_idx]

        rng = random.Random(seed if seed is not None else self.seed)
        return rng.choice(candidates)

    @staticmethod
    def hamming_distance(config1, config2):
        return sum(a != b for a, b in zip(config1, config2))

    @staticmethod
    def config_to_dict(config):
        return {name: int(state) for name, state in zip(ATTRIBUTE_NAMES, config)}

    @staticmethod
    def dict_to_config(d):
        return tuple(int(d[name]) for name in ATTRIBUTE_NAMES)
