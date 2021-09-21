import numpy as np
import n_sphere as ns
import warnings

from solve_lqr import lqr, dlqr
from scipy.integrate import solve_ivp


warnings.simplefilter('always', ImportWarning)

"""
from 
Scalable Traffic Models for Scheduling of
Linear Periodic Event-Triggered Controllers
Gabriel de A. Gleizer ∗ Manuel Mazo Jr. ∗
October 1, 2020
"""


class PETC8D():
    def __init__(self, n_traces, tmax, delta_t, sigma_tr, seed, noise=False):
        self.n_traces = n_traces
        self.tmax = tmax
        self.delta_t = delta_t
        self.sigma_trig = sigma_tr
        self.seed = seed
        self.noise_flag = noise
        self.nstd = np.sqrt(1e-3)
        self.n_xp = 8

        self.ap = np.array([
            [1.38, -0.208, 6.715, -5.676],
            [-0.581, -4.29, 0., 0.675],
            [1.067, 4.273, -6.654, 5.893],
            [0.048, 4.273, 1.343, -2.104]
        ])
        self.bp = np.array([
            [0., 0.],
            [5.679, 0.],
            [1.136, -3.146],
            [1.136, 0.]
        ])

        self.Ap = np.block([
            [self.ap, np.zeros_like(self.ap)],
            [np.zeros_like(self.ap), self.ap]
        ])

        self.Bp = np.block([
            [self.bp, np.zeros_like(self.bp)],
            [np.zeros_like(self.bp), self.bp]
        ])

        Q = np.eye(self.Ap.shape[0])
        R = 0.5*np.eye(self.Bp.shape[1])
        self.K, _, __ = lqr(self.Ap, self.Bp, Q, R)
        self.K = self.K

    def compute_trajectories_dist(self):
        """
        computes event-triggered controlled trajectories of the system of choice
        :return:
        """
        np.random.seed(self.seed)
        N_traces = self.n_traces
        init_conditions = 2.0 * np.random.randn(N_traces, 2*self.n_xp)
        init_conditions[:, self.n_xp:] = 0.
        radius = 1.0
        norms = np.sum(init_conditions**2, axis=1)[:, None]
        init_conditions = init_conditions/np.sqrt(norms)

        # init_conditions = np.hstack([radius * np.sin(phi), radius*np.cos(phi)])
        # init_conditions = np.zeros((N_traces, 2*self.n_xp))
        # init_conditions[:, 0:4] = 10.*np.array([[1., -1., -1., 1.]])
        all_inters_abs, all_inters_rel, xi_sampled_series, t_sampled_series = [], [], [], []
        xi_minus_one, t_minus_ones = [], []
        xi0_sampled_series = []
        len_sequences = []
        all_ys = []
        for idx, init_y in enumerate(init_conditions):
            xi0 = init_y.copy()
            ys, ts, t_sampled, xi_sampled = self.event_triggered_discrete_ist(xi0)
            if len(t_sampled) > 1:
                inters_abs, _ = self.compute_intersampling(t_sampled, ts, ys)
                inters_abs += [inters_abs[-1]]
                all_inters_abs += [inters_abs]
                # all_inters_rel += [inters_rel]
                xi_sampled_series += [xi_sampled]
                xi_minus_one += [xi_sampled[0]] + [xi_sampled[i] for i in range(len(xi_sampled)-1)]
                t_sampled_series += [t_sampled]
                xi0_sampled_series += [init_y] * len(xi_sampled)
                len_sequences.append(len(xi_sampled))
                all_ys.append(ys)

        separate_t_sampled = t_sampled_series
        separate_xi_sampled = [np.sum(np.array(x) ** 2, axis=1) for x in xi_sampled_series]
        t_sampled_series = np.concatenate(t_sampled_series)
        normalised_xi_sampled = [np.sum(np.array(xi_series) ** 2, axis=1) / np.sum(init_conditions[i, :] ** 2)
                                 for i, xi_series in enumerate(xi_sampled_series)]
        normalised_xi_sampled = np.concatenate(normalised_xi_sampled)
        single_xi_sampled = np.concatenate(xi_sampled_series)[:, 0:self.n_xp]
        single_xi_minus_one = np.stack(xi_minus_one)[:, 0:self.n_xp]
        xi_sampled_series = np.sum(np.concatenate(xi_sampled_series) ** 2, axis=1)
        xi_minus_one = np.sum(np.stack(xi_minus_one)**2, axis=1)
        observed = np.concatenate(all_inters_abs)
        # all_inters_rel = np.concatenate(all_inters_rel)
        single_xi0_sampled = np.array(xi0_sampled_series)
        xi0_sampled_series = np.sum(np.array(xi0_sampled_series) ** 2, axis=1)

        return observed, t_sampled_series, xi_sampled_series, single_xi_sampled, \
               xi0_sampled_series, single_xi0_sampled, xi_minus_one, single_xi_minus_one, \
               len_sequences, all_ys

    def event_triggered_discrete_ist(self, xi):
        """
        the intersampling time must be tau = h * delta_t
        :param xi:
        :param tmax:
        :param delta_t:
        :return:
        """

        # Define event function and make it a terminal event
        # def event(t, xi):
        #     # sigma * x ** 2 - err ** 2 = 0
        #     return sum(xi[8:] ** 2) - self.sigma_trig * sum(xi[0:8] ** 2)
        #
        # event.terminal = True
        # event.direction = 0

        def plant_dynamics(A, B, K, y):

            A_tot = np.block([[A + B @ K, B @ K], [-A - B @ K, -B @ K]])
            dydt = A_tot @ y

            # solve ivp wants a 1-d array
            if len(dydt.shape) > 1:
                dydt = np.array(dydt)[0]

            return dydt

        def error_dynamics(t, y, A, B, K):
            # state: xp, yp, xc, yc, e
            # error = x_t_k - x_t ==> x_t_k = e + x_t
            # x' = f(x, u(x_t_k))
            # e' = -f(x, u(x_t_k))
            xp_t_k = y[:8] + y[8:]

            dydt = plant_dynamics(A, B, K, y)

            return dydt

        tstart = 0.0
        sampl = int(self.tmax / self.delta_t) + 1
        t = np.linspace(0, self.tmax, sampl)
        ts = []
        ys = []
        t_sampled = [0.0]
        xi_sampled = [xi]
        # half = len(xi) // 2

        stop = False
        while not stop:
            sol = solve_ivp(error_dynamics, (tstart, self.tmax), xi,
                            args=(self.Ap, self.Bp, -self.K,),
                            t_eval=t, method='RK45')  #, max_step=0.01, min_step=0.001)
            if self.noise_flag:
                sol.y = np.random.normal(sol.y, scale=self.nstd)
            ts.append(sol.t)
            ys.append(sol.y)
            for idx, y in enumerate(ys[-1].T):
                # stop if reached the end
                if idx == ys[-1].shape[1] - 1:
                    stop = True
                # otherwise, check
                if sum(y[self.n_xp:] ** 2) - self.sigma_trig * sum(y[0:self.n_xp] ** 2) > 0.0:
                    tstart = ts[-1][idx]
                    t_sampled.append(tstart)
                    # Reset initial state
                    xi = y.copy()
                    xi[8] = 0.0
                    xi[9] = 0.0
                    xi[10] = 0.0
                    xi[11] = 0.0
                    xi[12] = 0.0
                    xi[13] = 0.0
                    xi[14] = 0.0
                    xi[15] = 0.0

                    xi_sampled.append(xi)
                    # Restrict t_eval
                    idx_to_cut = np.max(np.where((t <= tstart)))
                    t = t[idx_to_cut + 1:]
                    # clean ts and ys
                    idx_to_cut = np.max(np.where((ts[-1] <= tstart)))
                    ts[-1] = ts[-1][:idx_to_cut + 1]
                    ys[-1] = ys[-1][:, :idx_to_cut + 1]
                    break

        return np.hstack(ys), np.concatenate(ts), t_sampled, xi_sampled

    def compute_intersampling(self, t_sampled, ts, ys):
        # intersampling times
        intersample = [t_sampled[i + 1] - t_sampled[i] for i in range(len(t_sampled) - 1)]
        intersample_tuple, t = [], 0
        # for i in range(len(intersample)):
        #     modulo_x = np.sum(ys[:, np.where(ts == t)[0]][:, -1] ** 2)
        #     intersample_tuple.append( (t, modulo_x, intersample[i]) )
        #     t = t + intersample[i]

        return intersample, intersample_tuple

    def get_input_output(self, desired_components, stdize):
        components_x = []
        for comp in desired_components:
            if comp == 'time':
                components_x.append(self.t_sampled_series)
            elif comp == 'norm_xi':
                components_x.append(np.sqrt(self.xi_sampled_series)[:, None])
            elif comp == 'single_xi':
                for i in range(self.single_xi_sampled.shape[1]):
                    components_x.append(self.single_xi_sampled[:, i])
            elif comp == 'norm_xi0':
                components_x.append(np.sqrt(self.xi0_sampled_series))
            elif comp == 'single_xi0':
                for i in range(self.single_xi0_sampled.shape[1]):
                    components_x.append(self.single_xi0_sampled[:, i])
            elif comp == 'polar':
                # pc = z2polar(cart2z(self.single_xi_sampled[:, 0], self.single_xi_sampled[:, 1]))
                pc = ns.convert_spherical(self.single_xi_sampled)
                for idx, c in enumerate(pc.T):
                    # if idx > 0:
                    #     c = c % np.pi * np.prod(np.sign(self.single_xi_sampled), axis=1)
                    components_x.append(c)
            elif comp == 'angles':
                components_x.append(ns.convert_spherical(self.single_xi_sampled)[:, 1:])
                # components_x.append(np.arctan2(self.single_xi_sampled[:, 1], self.single_xi_sampled[:, 0]))

        # add x_minus_one and single_x_min_one after the std-isation
        if len(desired_components) == 1 and desired_components[0] != 'polar' and \
           desired_components[0] != 'single_xi0':
            x = components_x[0][:,None]
        else:
            x = np.hstack(components_x).T

        if stdize:
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            print('Info: Data are standardized.')
        else:
            warnings.warn('Info: Data are *not* standardized.', ImportWarning)

        if 'xi_min_one' in desired_components:
            mean = (np.sqrt(self.xi_sampled_series)).mean()
            std = (np.sqrt(self.xi_sampled_series)).std()
            xi_min_one = (np.sqrt(self.xi_min_one) - mean) / std
            x = np.hstack([x, xi_min_one[:,None]])
        if 'single_xi_min_one' in desired_components:
            for i in range(self.single_xi_min_one.shape[1]):
                # self.single_xi_min_one[:, i]
                mean = (self.single_xi_sampled[:, i]).mean()
                std = (self.single_xi_sampled[:,i]).std()
                std_xi_min_one = (self.single_xi_min_one[:,i] - mean) / std
                x = np.hstack([x, std_xi_min_one[:, None]])

        # standardize the intersampling times from 1 to N, all within integers
        y = np.round(self.observed / self.delta_t - 1, 3)
        y = np.array(y, dtype=int)

        return x, y

    def get_data(self, desired_components, stdize=False):
        """
        gets x and y based on desired components
        :param desired_components:
        :return:
        """
        self.observed, self.t_sampled_series, self.xi_sampled_series, self.single_xi_sampled, \
        self.xi0_sampled_series, self.single_xi0_sampled, self.xi_min_one, self.single_xi_min_one, \
        len_sequences, all_ys = self.compute_trajectories_dist()
        x, y = self.get_input_output(desired_components, stdize)
        return x, y, len_sequences, all_ys

    def get_trajectories(self):
        if self.observed is not None:
            return self.single_xi_sampled
        else:
            raise ValueError('Compute trajectories first!')