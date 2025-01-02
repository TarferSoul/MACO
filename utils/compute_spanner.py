import numpy as np

from env import Arm, SupArm


# ' @param X matrix of size (N, K)
# ' @param C approximation value
# ' @param quiet boolean
# ' @returns a C-approximate barycentric spanner of the columns of X as done in [Awerbuch et al., 2008]
def barycentric_spanner(X, C=1, quiet=True, precision=1e-6):
    N, K = X.shape
    assert K > 0

    def det_(M):
        sign, logdet = np.linalg.slogdet(M)
        return sign * np.exp(logdet)

    if (min(N, K) == K):
        return list(range(K))  # result from Awerbuch et al.
    # basis of set of arm features
    Fx = np.matrix(np.eye(N))
    F = [None] * N
    S = range(K)
    for a in range(N):
        other_ids = [u for u in range(N) if (u != a)]
        # replace Fx[:,a] with X[:,s], s in S
        max_det, max_det_id = -float("inf"), None
        for s in S:
            Xa = np.hstack((X[:, s].reshape((X.shape[0], 1)), Fx[:, other_ids]))
            dXa = det_(Xa)
            # keep it linearly independent
            if (dXa > max_det):
                max_det = dXa
                max_det_id = s
        Fx[:, a] = X[:, max_det_id].reshape((X.shape[0], 1))
        F[a] = max_det_id
    # transform basis into C-approximate barycentric spanner of size <= d
    done = False
    while (not done):
        found = False
        for s in S:
            for a in range(N):
                other_ids = [u for u in range(N) if (u != a)]
                det_Xs = det_(np.hstack((X[:, s].reshape((X.shape[0], 1)), Fx[:, other_ids])))  # |det(x, X_{-a})|
                det_Xa = det_(Fx)  # |det(X_a, X_{-a})|
                if ((det_Xs - C * det_Xa) > precision):  # due to machine precision, might loop forever otherwise
                    Fx[:, a] = X[:, s].reshape((X.shape[0], 1))
                    F[a] = s
                    found = True
        done = not found
    spanner = [f for f in F if (str(f) != F)]
    if (not quiet):
        print("Spanner size d = " + str(len(spanner)) + " | K = " + str(K)),
    return F


if __name__ == '__main__':
    dataset_list = ["movielens_25m", "movielens_25m_wrong", "lastfm", "yelp_pca", "yelp_svd"]

    for dataset in dataset_list:
        dataset_path = "input_data/" + dataset
        AM = Arm.ArmManager(dataset_path)
        AM.loadArms()
        print(f'[main] Finish loading arms: {AM.n_arms}')

        SAM = SupArm.SupArmManager(dataset_path, AM)
        SAM.loadArmSuparmRelation()
        print(f'[main] Finish loading suparms: {SAM.num_suparm}')

        suparm_matrix = np.zeros((SAM.num_suparm, AM.dim))
        suparm_index = {}
        for i, suparm in enumerate(SAM.suparms.values()):
            suparm_matrix[i, :] = suparm.fv.T
            suparm_index[i] = suparm.id

        spanner_index = barycentric_spanner(suparm_matrix.T, C=1, quiet=True, precision=1e-6)
        barycentric_spanner_list = [suparm_index[i] for i in spanner_index]

        f = open(dataset_path + '/saved_spanner.txt', 'w')
        f.write(str(barycentric_spanner_list))
