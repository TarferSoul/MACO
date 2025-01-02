import json

import numpy as np
from sklearn.preprocessing import normalize

from env import Arm, SupArm, User


def generate_info(U, V, user_num, top_items, attr_list, target_prefix):
    # user info: user_preference.txt
    f = open(target_prefix + '/user_preference.txt', 'w')
    for i in range(user_num):
        user = dict()
        user['uid'] = i
        user['preference_v'] = U[i].reshape((-1, 1)).tolist()
        f.write(json.dumps(user) + '\n')
    f.close()

    # pair items and attributes
    attr_list = [attr_list[i] if i in attr_list else [] for i in top_items]
    item_attr = dict()
    for attr in attr_list:
        for item in attr:
            if item not in item_attr:
                item_attr[item] = len(item_attr)

    # item info: arm_info.txt
    # item-attribute info: arm_suparm_relation.txt
    f = open(target_prefix + '/arm_suparm_relation.txt', 'w')
    g = open(target_prefix + '/arm_info.txt', 'w')
    separate = ','
    for i, attr in enumerate(attr_list):
        arm = {'a_id': i, 'fv': V[i, :].reshape((-1, 1)).tolist()}
        f.write(str(i) + '\t' + separate.join([str(item_attr[item]) for item in attr]) + ',\n')
        g.write(json.dumps(arm) + '\n')
    f.close()
    g.close()


if __name__ == '__main__':
    dataset_path = "input_data/syn_L_50"

    AM = Arm.ArmManager(dataset_path)
    AM.loadArms()
    print(f'[main] Finish loading arms: {AM.n_arms}')

    SAM = SupArm.SupArmManager(dataset_path, AM)
    SAM.loadArmSuparmRelation()
    print(f'[main] Finish loading suparms: {SAM.num_suparm}')

    UM = User.UserManager(dataset_path)
    UM.loadUser()
    print(f'[main] Finishing loading users: {UM.n_user}')

    target_prefix = "input_data/syn_L_50_normalized"
    f = open(target_prefix + '/user_preference.txt', 'w')
    for user in UM.users.values():
        entry = dict()
        entry['uid'] = user.uid
        entry['preference_v'] = normalize(user.theta.T).reshape((-1, 1)).tolist()
        f.write(json.dumps(entry) + '\n')
    f.close()

    f = open(target_prefix + '/arm_info.txt', 'w')
    separate = ','
    for arm_id in sorted(AM.arms.keys()):
        arm = {'a_id': arm_id, 'fv': normalize(AM.arms[arm_id].fv.T).reshape((-1, 1)).tolist()}
        f.write(json.dumps(arm) + '\n')
    f.close()
