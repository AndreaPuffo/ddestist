from sortedcontainers import SortedList
from itertools import chain
import numpy as np


class Partition():
    """
    Partition is an obj represents a partition in a R^n space
    has three attributes: lowers, uppers, ist
    lowers: list of lower bounds, one for every quantity involved in the partitioning
    uppers: list of upper bounds, one for every quantity involved in the partitioning
    ist: float/int, the inter-sample time associated to the partition
    """
    def __init__(self, sample, lowers, centers, uppers, ist):
        self.sample = sample
        self.lowers = np.array(lowers)
        self.centers = np.array(centers)
        self.uppers = np.array(uppers)
        self.ist = ist

        assert np.all(lowers < uppers)

    def __lt__(self, other):
        idx_to_compare = 0
        while idx_to_compare < len(self.lowers):
            if self.lowers[idx_to_compare] < other.lowers[idx_to_compare]:
                return True
            elif self.lowers[idx_to_compare] > other.lowers[idx_to_compare]:
                return False
            else:
                idx_to_compare += 1
        return None

    def get_ist(self):
        return self.ist

    def _set_ist(self, new_ist):
        self.ist = new_ist

    def get_bounds(self):
        return self.lowers, self.uppers


class Abstraction():
    """
    collection of Partition !sorted! objects (keep them sorted helps a quicker search)
    sorted by the lowers attribute of Partitions
    can add partition, remove partition
    split a partition
    """
    def __init__(self):
        self.partitions = None
        self.parts_dim = None

    # create or add to the abstraction form a list of lb, ub, ists
    def add_from_list(self, sample_list, lb_list, ct_list, ub_list, ist_list):
        """
        creates
        :param sample_list:
        :param lb_list:
        :param ct_list:
        :param ub_list:
        :param ist_list:
        :return:
        """
        assert len(lb_list) == len(ub_list) and len(ub_list) == len(ist_list) and len(sample_list) == len(lb_list)
        for idx in range(len(lb_list)):
            p = Partition(sample=sample_list[idx], lowers=lb_list[idx],
                          centers=ct_list[idx], uppers=ub_list[idx], ist=ist_list[idx])
            self.add_p(p)

    # add partition
    def add_p(self, p):
        """
        add a partition to the abstraction
        :param p:
        :return:
        """
        if not self.partitions:
            self.partitions = SortedList([p])
        else:
            self.partitions.add(p)

    def remove_p(self, p):
        """
        removes the partition from the abstraction
        :param p:
        :return:
        """
        self.partitions.remove(p)

    def find_index_p(self, p):
        """
        finds the partition index
        :param p:
        :return:
        """
        return self.partitions.index(p)

    def belongs_to(self, point):
        """
        find which partition point belongs to
        :param point:
        :return:
        """
        for p in self.partitions:
            if np.all(p.lowers <= point) and np.all(point < p.uppers):
                return p
        return None

    def compute_parts_dim(self):

        if not self.partitions:
            raise ValueError('There are no partitions in the Abstraction')

        part = self.partitions[0]
        self.parts_dim = part.uppers - part.lowers

    def find_neighbor(self, p):
        """
        given a partition p, find all its neighbors (in a geom sense)
        :param p: partition object
        :return: list
        """
        if not self.partitions:
            raise ValueError('There are no partitions in the Abstraction')

        # compute partitions dimensions if not done already
        if self.parts_dim is None:
            self.compute_parts_dim()

        neighbors = []
        # partitions are neighbors when "aligned" in one axis and their distance is lower than threshold
        for idx, part in enumerate(self.partitions):
            # check there are n-1 zeros
            ctr_distance = np.abs(part.centers - p.centers)
            if np.sum(ctr_distance == 0) == len(p.centers)-1:
                # find the dim where the distance is non zero
                idx_nnz = np.where(ctr_distance != 0)
                # to avoid numerical imprecision, add a small amount
                epsilon = np.min(self.parts_dim)/10
                # if the distance between the centers is <= than the required dim, add as neighbor
                if ctr_distance[idx_nnz] <= self.parts_dim[idx_nnz] + epsilon:
                    neighbors.append(idx)
        return neighbors

    def find_color_neighbor(self, p, idx_p):
        """
        given a partition p, find all its neighbors (in a geom sense) with the same ist (the 'color')
        :param p: partition object
        :param idx_p: partition index
        :return: list
        """
        if not self.partitions:
            raise ValueError('There are no partitions in the Abstraction')

        # compute partitions dimensions if not done already
        if self.parts_dim is None:
            self.compute_parts_dim()

        center_neighbors = []
        # candidate neighbors
        dims = len(self.parts_dim)
        n_parts = np.sqrt(len(self.partitions))

        neig_plus = [int(n_parts**i) for i in range(dims)]
        candidate_neighbors = idx_p + np.hstack([np.array(neig_plus), -np.array(neig_plus)])
        # saturate numbers between 0 and total number of partitions
        candidate_neighbors = np.maximum(0, np.minimum(len(self.partitions)-1, candidate_neighbors))
        # partitions are neighbors when "aligned" in one axis and their distance is lower than threshold
        for idx in candidate_neighbors:
            # get the partition
            part = self.partitions[idx]
            # check there are n-1 zeros
            ctr_distance = np.abs(part.centers - p.centers)
            if np.sum(ctr_distance == 0) == len(p.centers)-1:
                # find the dim where the distance is non zero
                idx_nnz = np.where(ctr_distance != 0)
                # to avoid numerical imprecision, add a small amount
                epsilon = np.min(self.parts_dim)/10
                # if the distance between the centers is <= than the required dim, add as neighbor
                if ctr_distance[idx_nnz] <= self.parts_dim[idx_nnz] + epsilon \
                        and p.ist == part.ist:
                    center_neighbors.append(idx)
        return center_neighbors

    def find_all_neighbors(self):
        """
        finds all neighbors of the partitions
        :return: list
        """
        all_n = []
        for p in self.partitions:
            all_n.append(self.find_neighbor(p))
        return all_n

    def find_colored_partitions(self):
        """
        splits the partitions based on the neighbors and the ists (the 'colors')
        :return: list of sets, list of sets
        #todo: think of a better return format
        """

        print('Finding color partitions')

        all_c_n = []
        c_n_idxonly = []
        for idx_p, p in enumerate(self.partitions):
            # colored neighbor index
            c_n_idx = self.find_color_neighbor(p, idx_p)
            # add partition itself
            c_n_idx.append(idx_p)
            # actual partitions and index
            c_n = [(self.partitions[i], i) for i in c_n_idx]

            set_cn = set(c_n)
            idx_to_add = np.where([(p, idx_p) in sublist for sublist in all_c_n])

            if len(idx_to_add[0]) == 0:
                all_c_n.append(set_cn)
                c_n_idxonly.append(set(c_n_idx))
            else:
                all_c_n[idx_to_add[0][0]].update(set_cn)
                c_n_idxonly[idx_to_add[0][0]].update(set(c_n_idx))

        print('Sorting them')

        # possible define a new function
        #
        # in case of two detached partitions that should be one, check and merge
        final_partition_and_index, temp_p_idx = [], all_c_n.copy()
        final_index, temp_index = [], c_n_idxonly.copy()
        sum_partitions = np.sum([len(sublist) for sublist in c_n_idxonly])
        sum_partitions_old = 0
        while sum_partitions > len(self.partitions):
            i = 0
            if sum_partitions_old == sum_partitions:
                temp_p_idx, temp_index = self.reshuffle_partitions(temp_p_idx, temp_index)
            while i < len(temp_index)-1:
                # union of sets
                unity = temp_index[i].union(temp_index[i+1])
                union_part_index = temp_p_idx[i].union(temp_p_idx[i+1])
                # if the length of the union is less than the sum of the two,
                # there is a common elem
                # -> merge the two
                found_common = len(unity) < len(temp_index[i]) + len(temp_index[i+1])
                if len(unity) < len(temp_index[i]) + len(temp_index[i+1]):
                    final_index.append(unity)
                    final_partition_and_index.append(union_part_index)
                    i = i+2
                    if i == len(temp_index) - 1:
                        final_index.append(temp_index[i])
                        final_partition_and_index.append(temp_p_idx[i])
                else:
                    final_index.append(temp_index[i])
                    # final_index.append(temp_index[i+1])
                    final_partition_and_index.append(temp_p_idx[i])
                    # final_partition_and_index.append(temp_p_idx[i+1])
                    i = i + 1
                    if i == len(temp_index) - 1:
                        final_index.append(temp_index[i])
                        final_partition_and_index.append(temp_p_idx[i])

            # random permutation of the lists
            temp_p_idx = final_partition_and_index.copy()
            temp_index = final_index.copy()
            sum_partitions_old = sum_partitions
            sum_partitions = np.sum([len(sublist) for sublist in final_index])
            final_partition_and_index, final_index = [], []

        assert sum_partitions == len(self.partitions)
        return temp_p_idx, temp_index

    def reshuffle_partitions(self, parts_and_index, indexes):
        """
        reshuffles lists
        :param parts_and_index:
        :param indexes:
        :return:
        """

        perm = np.random.permutation(range(len(indexes)))
        permuted_parts = [parts_and_index[i] for i in perm]
        permuted_indxs = [indexes[i] for i in perm]

        return permuted_parts, permuted_indxs

    def split_partition(self, p, candidate):
        """
        splits the partition p in two along the longest bound's axis
        :param p: Partition object
        :param candidate: (point, ist)
        :return:
        """
        # get candidate's point and ist
        point, new_ist = candidate[0], candidate[1]
        # remove partition from abstraction
        self.remove_p(p)
        # find the longest axis and split it
        max_bound = -1.0
        for dim in range(len(p.lowers)):
            l = np.abs(p.sample[dim] - point[dim])
            if l > max_bound:
                max_bound = l
                dim_bound = dim
        # create new bounds
        lowers_left, lowers_right = p.lowers.copy(), p.lowers.copy()
        uppers_left, uppers_right = p.uppers.copy(), p.uppers.copy()
        new_dimension = (point[dim_bound]+p.sample[dim_bound])/2
        lowers_right[dim_bound] = new_dimension
        uppers_left[dim_bound] = lowers_right[dim_bound]

        # find which partition the candidate belongs to
        if np.all(lowers_left <= point) and np.all(point < uppers_left):
            # create partitions w/ the old ist
            p1 = Partition(point, lowers_left, uppers_left, new_ist)
            p2 = Partition(p.sample, lowers_right, uppers_right, p.get_ist())
        elif np.all(lowers_right <= point) and np.all(point < uppers_right):
            p1 = Partition(p.sample, lowers_left, uppers_left, p.get_ist())
            p2 = Partition(point, lowers_right, uppers_right, new_ist)
        else:
            print('*Point not in partition*')
            ValueError('The point doe not belong to the partition!')
        # add the partitions to the abstraction
        self.add_p(p1)
        self.add_p(p2)
