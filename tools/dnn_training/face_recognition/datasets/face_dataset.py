import os

import torch
from PIL import Image

from torch.utils.data import Dataset


LFW_OVERLAPPED_VGGFACE2_CLASS_NAMES = ['n000021', 'n000137', 'n000172', 'n000184', 'n000195', 'n000199', 'n000220', 'n000242', 'n000255', 'n000272', 'n000281', 'n000297', 'n000310', 'n000359', 'n000373', 'n000379', 'n000402', 'n000420', 'n000427', 'n000429', 'n000483', 'n000560', 'n000562', 'n000568', 'n000645', 'n000667', 'n000692', 'n000709', 'n000712', 'n000755', 'n000780', 'n000810', 'n000816', 'n000817', 'n000835', 'n000887', 'n000888', 'n000902', 'n000906', 'n000912', 'n000946', 'n000947', 'n001042', 'n001048', 'n001051', 'n001054', 'n001057', 'n001060', 'n001091', 'n001095', 'n001096', 'n001105', 'n001106', 'n001110', 'n001111', 'n001113', 'n001142', 'n001143', 'n001155', 'n001156', 'n001159', 'n001171', 'n001200', 'n001228', 'n001235', 'n001243', 'n001257', 'n001288', 'n001309', 'n001329', 'n001360', 'n001367', 'n001379', 'n001387', 'n001419', 'n001434', 'n001437', 'n001473', 'n001478', 'n001538', 'n001540', 'n001547', 'n001548', 'n001551', 'n001567', 'n001568', 'n001595', 'n001607', 'n001642', 'n001656', 'n001663', 'n001695', 'n001698', 'n001700', 'n001727', 'n001775', 'n001779', 'n001781', 'n001795', 'n001805', 'n001813', 'n001825', 'n001839', 'n001873', 'n001874', 'n001934', 'n001962', 'n001968', 'n001971', 'n001986', 'n002017', 'n002018', 'n002019', 'n002025', 'n002028', 'n002038', 'n002056', 'n002057', 'n002067', 'n002081', 'n002094', 'n002120', 'n002133', 'n002135', 'n002141', 'n002173', 'n002178', 'n002227', 'n002244', 'n002251', 'n002259', 'n002269', 'n002272', 'n002274', 'n002278', 'n002298', 'n002310', 'n002316', 'n002342', 'n002356', 'n002360', 'n002388', 'n002391', 'n002413', 'n002452', 'n002454', 'n002460', 'n002465', 'n002471', 'n002478', 'n002494', 'n002497', 'n002537', 'n002547', 'n002572', 'n002574', 'n002595', 'n002630', 'n002654', 'n002666', 'n002673', 'n002684', 'n002725', 'n002759', 'n002782', 'n002814', 'n002844', 'n002854', 'n002868', 'n002947', 'n002969', 'n002982', 'n003013', 'n003020', 'n003022', 'n003039', 'n003098', 'n003101', 'n003147', 'n003186', 'n003198', 'n003205', 'n003206', 'n003212', 'n003228', 'n003257', 'n003266', 'n003303', 'n003311', 'n003353', 'n003359', 'n003360', 'n003363', 'n003383', 'n003385', 'n003391', 'n003392', 'n003422', 'n003426', 'n003474', 'n003476', 'n003523', 'n003619', 'n003623', 'n003628', 'n003648', 'n003671', 'n003685', 'n003689', 'n003709', 'n003712', 'n003724', 'n003728', 'n003749', 'n003756', 'n003767', 'n003785', 'n003803', 'n003806', 'n003816', 'n003817', 'n003820', 'n003866', 'n003876', 'n003879', 'n003880', 'n003885', 'n003904', 'n003908', 'n003912', 'n003937', 'n003946', 'n003952', 'n003969', 'n003972', 'n003973', 'n003974', 'n003979', 'n003982', 'n004014', 'n004020', 'n004030', 'n004043', 'n004081', 'n004083', 'n004084', 'n004106', 'n004132', 'n004144', 'n004150', 'n004151', 'n004154', 'n004181', 'n004197', 'n004198', 'n004204', 'n004216', 'n004218', 'n004220', 'n004231', 'n004236', 'n004263', 'n004327', 'n004328', 'n004375', 'n004407', 'n004408', 'n004429', 'n004439', 'n004451', 'n004457', 'n004480', 'n004481', 'n004488', 'n004580', 'n004588', 'n004617', 'n004644', 'n004646', 'n004651', 'n004652', 'n004661', 'n004674', 'n004703', 'n004715', 'n004720', 'n004737', 'n004756', 'n004781', 'n004782', 'n004841', 'n004844', 'n004861', 'n004893', 'n004902', 'n004905', 'n004949', 'n004990', 'n005009', 'n005026', 'n005035', 'n005053', 'n005054', 'n005056', 'n005057', 'n005060', 'n005098', 'n005104', 'n005106', 'n005110', 'n005116', 'n005123', 'n005136', 'n005140', 'n005159', 'n005196', 'n005203', 'n005204', 'n005206', 'n005208', 'n005212', 'n005219', 'n005226', 'n005227', 'n005248', 'n005292', 'n005400', 'n005423', 'n005491', 'n005531', 'n005588', 'n005642', 'n005659', 'n005662', 'n005666', 'n005690', 'n005726', 'n005737', 'n005745', 'n005761', 'n005762', 'n005770', 'n005815', 'n005831', 'n005836', 'n005846', 'n005874', 'n005933', 'n005946', 'n005948', 'n005949', 'n005950', 'n005983', 'n005985', 'n005988', 'n006003', 'n006004', 'n006008', 'n006013', 'n006015', 'n006047', 'n006048', 'n006055', 'n006061', 'n006073', 'n006083', 'n006103', 'n006110', 'n006111', 'n006120', 'n006155', 'n006158', 'n006160', 'n006221', 'n006234', 'n006236', 'n006246', 'n006260', 'n006348', 'n006350', 'n006359', 'n006363', 'n006374', 'n006380', 'n006384', 'n006386', 'n006395', 'n006408', 'n006414', 'n006512', 'n006592', 'n006604', 'n006662', 'n006691', 'n006695', 'n006696', 'n006699', 'n006716', 'n006720', 'n006784', 'n006793', 'n006827', 'n006840', 'n006846', 'n006860', 'n006868', 'n006910', 'n006913', 'n006922', 'n006928', 'n006933', 'n006935', 'n006966', 'n006969', 'n006970', 'n007009', 'n007024', 'n007075', 'n007122', 'n007123', 'n007157', 'n007166', 'n007188', 'n007190', 'n007213', 'n007226', 'n007242', 'n007277', 'n007300', 'n007320', 'n007326', 'n007331', 'n007352', 'n007356', 'n007357', 'n007366', 'n007383', 'n007389', 'n007399', 'n007402', 'n007412', 'n007428', 'n007461', 'n007476', 'n007493', 'n007501', 'n007510', 'n007565', 'n007574', 'n007581', 'n007584', 'n007595', 'n007596', 'n007613', 'n007616', 'n007620', 'n007623', 'n007645', 'n007679', 'n007704', 'n007712', 'n007787', 'n007794', 'n007801', 'n007847', 'n007901', 'n007974', 'n007980', 'n007981', 'n008018', 'n008065', 'n008079', 'n008127', 'n008162', 'n008166', 'n008180', 'n008205', 'n008215', 'n008239', 'n008242', 'n008272', 'n008277', 'n008316', 'n008333', 'n008347', 'n008428', 'n008431', 'n008446', 'n008466', 'n008496', 'n008524', 'n008545', 'n008551', 'n008556', 'n008557', 'n008568', 'n008575', 'n008592', 'n008614', 'n008638', 'n008646', 'n008658', 'n008659', 'n008679', 'n008690', 'n008707', 'n008708', 'n008725', 'n008782', 'n008791', 'n008814', 'n008839', 'n008859', 'n008865', 'n008893', 'n008902', 'n008917', 'n008927', 'n008999', 'n009008', 'n009050', 'n009139', 'n009179', 'n009283']
LFW_OVERLAPPED_MSCELEB1M_CLASS_NAMES = ['m.03t4cz', 'm.03p4zn', 'm.01341q', 'm.023066', 'm.01yynm', 'm.07s7wk', 'm.05y82p', 'm.01d_7n', 'm.0224hd', 'm.0245tw', 'm.03mkck', 'm.05_70r', 'm.023phr', 'm.02vsp', 'm.0586vd', 'm.018ygt', 'm.05j0zz', 'm.013b7z', 'm.02tpys', 'm.014kb2', 'm.084dwc', 'm.01bnr9', 'm.02qzsv', 'm.05gttk', 'm.03qsqt', 'm.06cjq3', 'm.01y_rh', 'm.0gskh7', 'm.01fkqs', 'm.0261k6z', 'm.0c01c', 'm.03q1tm', 'm.03jd7z', 'm.068ggb', 'm.01cp_j', 'm.0347xl', 'm.03ds83', 'm.031h9y', 'm.03fqjd', 'm.01p9wn', 'm.0j28l1t', 'm.01tr5l', 'm.03lhhq', 'm.035m7s', 'm.09l7gt', 'm.0264h51', 'm.01kzy3', 'm.0bxr_8', 'm.048g68', 'm.023v5v', 'm.01t2h2', 'm.01lmd6', 'm.05n8_wc', 'm.0f4vbz', 'm.0pdcdhz', 'm.0ftfrr', 'm.02x5q_k', 'm.01vhb0', 'm.02cgqp', 'm.05p1y0r', 'm.036wtl', 'm.01p_kk', 'm.01yym5', 'm.012bk', 'm.0204ym', 'm.0ckpyv', 'm.051qh7', 'm.029985', 'm.082hs2', 'm.04fbt1', 'm.0fmpmx', 'm.0pstz', 'm.011p3', 'm.02l3b1', 'm.022wp3', 'm.025wkbs', 'm.021m32', 'm.02pr_2h', 'm.04rb3d', 'm.044khm', 'm.0b742yn', 'm.09y_q5', 'm.098sq7', 'm.02n670', 'm.013tcv', 'm.01nbjr', 'm.01bhq8', 'm.058rsb', 'm.06_1s3', 'm.04l118', 'm.02t244', 'm.02843j', 'm.06plcb', 'm.023lqp', 'm.01qkdh', 'm.09_9nl', 'm.01mpq7s', 'm.05214n', 'm.01jjdg7', 'm.0b_j2', 'm.02_fs7', 'm.05nbg1', 'm.07zk65', 'm.0154r3', 'm.028t43', 'm.017dzj', 'm.01_pb_', 'm.01sd9r', 'm.01s7zw', 'm.04lj06', 'm.0gfgzcp', 'm.01ym3s', 'm.03bnrg', 'm.01qwpt', 'm.01kyns', 'm.06f091', 'm.0hyrg', 'm.05dhdz', 'm.024k6h', 'm.01l87db', 'm.05xc43', 'm.05chnj', 'm.012gq6', 'm.08rt4p', 'm.0518nv', 'm.02c6bt', 'm.031pbz', 'm.01krs', 'm.01dzl3', 'm.0hr3h8d', 'm.01t_mn', 'm.0fvszq', 'm.03qfp6', 'm.02vy8cg', 'm.05p8_y2', 'm.0g5sm_y', 'm.0hhxv91', 'm.03vttl', 'm.053nn8', 'm.03fv56', 'm.0cn6gm', 'm.0bx_q', 'm.01kk0s', 'm.0801_6', 'm.06njqy', 'm.023s8', 'm.0bksh', 'm.04y3tl', 'm.04v16_', 'm.0d7_4', 'm.031dns', 'm.03cvx8g', 'm.01bzhh', 'm.06k68k', 'm.01xh6j', 'm.0pnf3', 'm.049dy4y', 'm.0d06qs', 'm.01l1ls', 'm.019k2w', 'm.01kj0p', 'm.019n7x', 'm.03tt0_', 'm.013knm', 'm.01k7lw', 'm.0fswr7', 'm.01cwhp', 'm.03yjtb', 'm.031513', 'm.020ymy', 'm.02z0ck', 'm.0yc8l', 'm.053f8h', 'm.0411ytm', 'm.01l9p', 'm.04fkdr', 'm.01chwz', 'm.05tcsz2', 'm.0c4b30', 'm.0hwmx', 'm.07nj7w', 'm.07vyft', 'm.063f7w', 'm.01ybxf', 'm.019qv6', 'm.01cp71', 'm.0dpqlq', 'm.05m53h', 'm.06ml5k', 'm.04cf09', 'm.016_mj', 'm.01900g', 'm.0g8mk1', 'm.0260ct', 'm.0bfwxz', 'm.08z9xg', 'm.05vxct', 'm.07dn76', 'm.027m7j3', 'm.0253pn', 'm.05spws', 'm.01l9pz', 'm.027nx2m', 'm.0cc58j_', 'm.0211tw', 'm.05x2hd', 'm.0f41k8', 'm.027d12t', 'm.0421x9', 'm.017tn4', 'm.03mfqm', 'm.02p1q9', 'm.0bqs56', 'm.03c30n', 'm.01pybj', 'm.011zfz', 'm.069_ks', 'm.05myt24', 'm.083pq2', 'm.01r3ct', 'm.015c2f', 'm.09wjvn', 'm.06zkr3z', 'm.04d9yz', 'm.04t33m', 'm.0751tg', 'm.04_x29', 'm.01dhpj', 'm.06qv4y', 'm.0jl26', 'm.02fhw', 'm.01ctnp', 'm.03wyyk', 'm.02bh9', 'm.0205dx', 'm.0htcn', 'm.04pn7c', 'm.051z8k', 'm.01r7wc', 'm.01s4kt', 'm.02v0hy', 'm.02ghgl', 'm.040jzf', 'm.05p36x1', 'm.02hh_y', 'm.025ycc', 'm.01mh7tb', 'm.0fcmr1', 'm.034bb7', 'm.025r7k', 'm.03sqfv', 'm.046xmg', 'm.0138zs', 'm.02ct_k', 'm.01my4f', 'm.03gbkk', 'm.02cgqp', 'm.02dlfh', 'm.0dql67', 'm.043ftz', 'm.05wgvhp', 'm.08w95x', 'm.0cjng5', 'm.06j0cg', 'm.04jw71', 'm.02_75_', 'm.07_ttx', 'm.019sj8', 'm.0pmhf', 'm.01t3s_', 'm.0822l3', 'm.037bzt', 'm.01gqws', 'm.04mxvcc', 'm.0d0vj4', 'm.07_v3p', 'm.01wsj06', 'm.073y_j', 'm.06p39_', 'm.01vt9p3', 'm.08_nmx', 'm.0260zf', 'm.02f1c', 'm.01kwld', 'm.04_vq_', 'm.015yb0', 'm.0cp14lp', 'm.01v3vb', 'm.01mbwlb', 'm.06j02v', 'm.078q1l', 'm.03gbqk', 'm.01_j71', 'm.0ks3w3', 'm.069ggg', 'm.01r4ft', 'm.03mswq', 'm.04fhqh', 'm.06v1ms', 'm.09gc_wk', 'm.0bl2g', 'm.02tlx1', 'm.06pl0j', 'm.06nmdb', 'm.0235mg', 'm.027nt8', 'm.04dr0x', 'm.02r7jn', 'm.01bzhh', 'm.01zkk1', 'm.01515w', 'm.02502p', 'm.01qrf2', 'm.0411lx', 'm.0267d_', 'm.0ccqd7', 'm.03wgt48', 'm.01jb26', 'm.04sj1_', 'm.01gbbz', 'm.05np4c', 'm.03jq8c', 'm.01w3fc0', 'm.01ksss', 'm.012sk_', 'm.022wf_', 'm.085qrf', 'm.05g7t5', 'm.018pj3', 'm.03d7ykg', 'm.0159h6', 'm.0cjbl1s', 'm.03m83s', 'm.0jbv8', 'm.02vn3vm', 'm.027kn0', 'm.01jhw2', 'm.09jgcv', 'm.07f3t6', 'm.04lvys', 'm.04smkr', 'm.03hjrr', 'm.02nxk', 'm.01r9lx', 'm.026fb9', 'm.0cg81d', 'm.03z5hz', 'm.0263mk', 'm.02c_q_', 'm.01zzy_', 'm.08mm_5', 'm.08ckfm', 'm.03c44y', 'm.03m83s', 'm.05241j0', 'm.03v68d', 'm.01pt2r', 'm.0k269', 'm.033rq', 'm.07k4g7g', 'm.02pk0t2', 'm.03rlj6', 'm.03c8zx', 'm.022vdk', 'm.04ykxz', 'm.02wy12', 'm.03q5dr', 'm.0fqm7g0', 'm.0dvv9s', 'm.05d0sm', 'm.06n_nn', 'm.02vyw', 'm.08sb57', 'm.03dhy8', 'm.0412t6', 'm.03m7rkt', 'm.05khsy', 'm.01gtsh', 'm.036g2b', 'm.04k1kk', 'm.01q7cb_', 'm.0ks957', 'm.02xbw2', 'm.02pt11', 'm.07tmq9', 'm.037w1', 'm.03kn29', 'm.0d_hr', 'm.04bgs7', 'm.0344jy', 'm.07s_31', 'm.038p7s', 'm.0166w_', 'm.0c6vjs', 'm.020sr8', 'm.0343h', 'm.01c_xx', 'm.0191bn', 'm.034ls', 'm.037w7r', 'm.014jy6', 'm.027c96m', 'm.0c9hm', 'm.03mt9', 'm.0d_9f1', 'm.04l9wz', 'm.051ztf', 'm.01l1hr', 'm.06v748', 'm.03b0n5', 'm.05k13g', 'm.0502rv', 'm.03mfjm', 'm.038zc', 'm.0cwtm', 'm.05tbhj', 'm.01lykw', 'm.074pxw', 'm.016k38', 'm.080p_h', 'm.0jwwgnw', 'm.09z0lc', 'm.03pt18', 'm.0p81w', 'm.05lqps', 'm.0bnyhq', 'm.01qwly', 'm.01q4s5', 'm.03g4bf', 'm.02l6dy', 'm.03g0p', 'm.03h1tqw', 'm.02xjlj', 'm.02qppnm', 'm.01g1lp', 'm.016fnb', 'm.0p921', 'm.067tsz', 'm.02569p', 'm.0kxrb', 'm.036df9', 'm.072jm3', 'm.0402rg', 'm.024wt7', 'm.07flkk', 'm.0288vd8', 'm.02l1g0', 'm.05hj_k', 'm.09qvwf', 'm.02wwj9', 'm.05bm10', 'm.047m111', 'm.01vysy8', 'm.04n1l2', 'm.0flspx', 'm.03hjhb4', 'm.08l6td', 'm.09hnb', 'm.02bcgg', 'm.03pg42', 'm.02gjzj', 'm.0261rs', 'm.02w_xk', 'm.01b1gs', 'm.04d1cq', 'm.0d0gzz', 'm.036m8s', 'm.0cmmy9', 'm.05czcv', 'm.03935p', 'm.0f5zj6', 'm.04hrhm', 'm.04jg9ff', 'm.02r5h2t', 'm.0230wx', 'm.0bpr4y', 'm.0dgfsk', 'm.03y23jt', 'm.0jb54', 'm.0182q2', 'm.019d8_', 'm.019xyd', 'm.014jb8', 'm.01pwrjt', 'm.0cv34x', 'm.01xdlx', 'm.05l97n', 'm.06_69f', 'm.03qspn', 'm.01fq0x', 'm.0k2hrth', 'm.0bgjc8', 'm.02348n', 'm.01k5tg', 'm.01770r', 'm.09xg8', 'm.03bzz8t', 'm.066xc8', 'm.03c5bz', 'm.03v3j_', 'm.08jj7t', 'm.03d_03', 'm.06ln1j', 'm.0ksv3d', 'm.01z7_f', 'm.07fcvr', 'm.02751h2', 'm.04_vgv', 'm.034bg5', 'm.03crtr', 'm.039gzc', 'm.0chhjz', 'm.01rn_x', 'm.01qr1_', 'm.0ckb3y', 'm.06tc98', 'm.0245wb', 'm.04flrx', 'm.02g87m', 'm.079nvf', 'm.03syvx', 'm.0gprt0', 'm.0477_', 'm.0hvby', 'm.0231bb', 'm.081vxj', 'm.0m8_v', 'm.015ybz', 'm.03gbf7', 'm.01g7hs', 'm.033hb0', 'm.0459k', 'm.03kth6', 'm.02rq9n', 'm.04n0lkh', 'm.03bmvc', 'm.0320cg', 'm.01gvv5', 'm.0dst4x', 'm.03jjzf', 'm.023pzh', 'm.046l2', 'm.041xfx', 'm.02cnq1', 'm.0240vt', 'm.056svc', 'm.02h73f', 'm.0_5w6', 'm.05ty1w', 'm.0hd1l', 'm.0s8tynj', 'm.09zvcg', 'm.01m54p', 'm.04rxwx', 'm.0270jd', 'm.02238b', 'm.0684tm', 'm.03qhcnp', 'm.03ldpc', 'm.01kvrj', 'm.0fc34h', 'm.03h979', 'm.014ptb', 'm.07r_sc', 'm.027tcmx', 'm.01_8rq', 'm.056mxl', 'm.013vtr', 'm.05db50', 'm.02w5_k', 'm.07p160', 'm.08cfkh', 'm.050p2t', 'm.02n45z', 'm.0408r5', 'm.026l37', 'm.05cnkm', 'm.06x328', 'm.01wgsvv', 'm.01cbjz', 'm.046c6', 'm.036z2s', 'm.044h4', 'm.01sthx', 'm.02fp95', 'm.0150p7', 'm.01s4ss', 'm.036_x3', 'm.04yw4x', 'm.04yj5z', 'm.079m5z', 'm.01gkbj', 'm.017r13', 'm.0bymv', 'm.0pyqh', 'm.0208bk', 'm.02wvnnx', 'm.05yhv', 'm.0336gg', 'm.0gg6xn1', 'm.03ms0t', 'm.0l65n', 'm.0h653gg', 'm.02vxmw3', 'm.01xcly', 'm.03sfbh', 'm.0fyf5g', 'm.03ph6k', 'm.098hm1', 'm.0h7f2f', 'm.0jgvf', 'm.01z1ws', 'm.04vkvw', 'm.03whg42', 'm.0kc54', 'm.0bbwlp5', 'm.021v2z', 'm.0b16s0', 'm.01r93l', 'm.0lpjn', 'm.01frrf', 'm.02f91x', 'm.03npb_', 'm.0509bl', 'm.082r6n', 'm.02g0mx', 'm.01q9m5', 'm.0d5cy', 'm.02b6km', 'm.03ghnx', 'm.0gx02bb', 'm.0182qx', 'm.0b6m063', 'm.04n0yqn', 'm.0fq2760', 'm.02f95t', 'm.01f8ld', 'm.065h1p', 'm.019ncn', 'm.02050j', 'm.03bdbl', 'm.06bxgv', 'm.037721', 'm.02_bdt', 'm.01pfh3w', 'm.05zrbnd', 'm.01yk06', 'm.049l7', 'm.0127m7', 'm.04s6ts', 'm.05qrgk', 'm.0273c_', 'm.0jt9z', 'm.048lv', 'm.0235t5', 'm.03bzdvy', 'm.06cpj9', 'm.069nbk', 'm.01p85y', 'm.013yvd', 'm.047sth0', 'm.015tp7', 'm.01cy1c', 'm.01kwlwp', 'm.04fzk', 'm.02pl46g', 'm.0498f', 'm.03cmk_', 'm.0155zp', 'm.03p4qd', 'm.04crpl', 'm.02pj01j', 'm.069zyg', 'm.04znp2', 'm.015bw2', 'm.0337vz', 'm.04fhqh', 'm.04my5mx', 'm.08047z', 'm.04r6kn', 'm.045qln', 'm.0dpt2x', 'm.09j_2f', 'm.03qc9cc', 'm.027ypcs', 'm.04g8d', 'm.020_95', 'm.0krz6v', 'm.047n7c', 'm.0dfphs', 'm.014gf8', 'm.012tmz', 'm.03nzr_', 'm.01qx13', 'm.078dv3', 'm.01fkxr', 'm.09l7gt', 'm.04q4l', 'm.01pxrx', 'm.043ls76', 'm.05yqrl', 'm.03h_x0', 'm.04sh_1', 'm.0dvmd', 'm.021yhj', 'm.04svb2', 'm.04gsvmw', 'm.088_bk', 'm.02nwxc', 'm.05zv32', 'm.011_3s', 'm.0g9y1p7', 'm.0347ls', 'm.0byhnl', 'm.08qvjx', 'm.0gkxg7b', 'm.064r8yv', 'm.0qlry', 'm.026qnkm', 'm.03d70n0', 'm.0g476', 'm.01n048', 'm.01pfdg', 'm.0qs96j4', 'm.05dd_l', 'm.01rsdq', 'm.01270s', 'm.0p_2r', 'm.02lf70', 'm.09gcs', 'm.05q2n2', 'm.03fqzp', 'm.01zm1l', 'm.05_xn9', 'm.05dztx', 'm.01fhst', 'm.01fdc0', 'm.04_0sw4', 'm.01vs_v8', 'm.0139q5', 'm.01p8qv', 'm.043jp1_', 'm.02fzd3', 'm.026tmp', 'm.02wxbt', 'm.01bqmg', 'm.08d65g', 'm.03s4j_', 'm.01wv9p', 'm.03th34', 'm.044gjv', 'm.04cp9x', 'm.06jrvl', 'm.02kt6r', 'm.06l8jw', 'm.047dbx9', 'm.0mgb9', 'm.0kb3n', 'm.05zsgz', 'm.0c06sr', 'm.0375zc', 'm.0b3cs1', 'm.033h2v', 'm.0jwyq2x', 'm.09g82r', 'm.018phr', 'm.05ly3r', 'm.01mqnr', 'm.05jc7m', 'm.0b6z2c', 'm.0gmjf8', 'm.0jfzc', 'm.01v1ys3', 'm.024xmf', 'm.01msrs', 'm.019tyn', 'm.057hz', 'm.01bffy', 'm.09dh7n', 'm.03vjl6', 'm.05_tcb', 'm.026r8q', 'm.01rrd4', 'm.082vtn', 'm.028wtc', 'm.0dzyp6', 'm.022q61', 'm.05n_d_', 'm.0227p5', 'm.01jglh', 'm.04rdhn', 'm.0lgsq', 'm.080sy1', 'm.052hl', 'm.04g09qn', 'm.05fj6r', 'm.04bktz', 'm.0gvr2mx', 'm.043mw84', 'm.01j5ws', 'm.01r_drv', 'm.01fdpj', 'm.070j61', 'm.02r1qhb', 'm.0h3n5yz', 'm.054bt3', 'm.053n8s', 'm.03p5vy', 'm.04fw8yd', 'm.06qy4p', 'm.027f6w', 'm.04jfgvr', 'm.02_0bm', 'm.0263y0', 'm.0ck7wj', 'm.02722_', 'm.05c8f_', 'm.03qp6s', 'm.03lr3z', 'm.01dtxd', 'm.038m0d', 'm.0265bkr', 'm.01zgyp', 'm.0294fd', 'm.01phtd', 'm.07nb54', 'm.05s74y', 'm.04ffv_', 'm.09gh9sl', 'm.01728m', 'm.09r_jb', 'm.0k3mtc8', 'm.081xj6', 'm.02vmmz', 'm.02dwq5', 'm.03dpm5', 'm.07cktb', 'm.0641q5', 'm.03kml6', 'm.01_x4x', 'm.0296q2', 'm.03nw4q', 'm.02y_4xw', 'm.0346l4', 'm.032bfz', 'm.02dvwl', 'm.0d9mt_', 'm.01nxzv', 'm.03cymlv', 'm.043q6n_', 'm.01hhd7', 'm.018fzs', 'm.02s5m3', 'm.0lkr7', 'm.024zc8', 'm.0f9dpr', 'm.032kyt', 'm.02ps9k', 'm.07bp3c', 'm.08f2m4', 'm.05zwl4', 'm.0251xd', 'm.01t6xz', 'm.01rjfj', 'm.069lpq', 'm.05dgpl', 'm.02jxnz', 'm.01jhtj', 'm.0krnw', 'm.0ft68', 'm.016mbz', 'm.09rcjg', 'm.05kfs', 'm.059zx3', 'm.01zz8t', 'm.034g1z', 'm.06l0c3', 'm.016mj4', 'm.07s8p52', 'm.06w7jl9', 'm.01q_ph', 'm.03ghnx', 'm.05r5w', 'm.01cygd', 'm.051j_m', 'm.0615j_', 'm.0c3yd_j', 'm.0gt1yv', 'm.0b0mpg', 'm.05sy8', 'm.026qhcv', 'm.059g24', 'm.01f9yn', 'm.0czcd7c', 'm.01h75w', 'm.02l2s2', 'm.09s3gm', 'm.01gzfn', 'm.012nry', 'm.01kb6l', 'm.0drhp3', 'm.0206p0', 'm.037d35', 'm.01nzww', 'm.03q35x', 'm.0g58864', 'm.010ngb', 'm.038786', 'm.03bghb', 'm.067g_', 'm.01nw3d', 'm.061s_', 'm.0p3q3', 'm.028b5vn', 'm.0d8kxt', 'm.01lghn', 'm.07s6zxp', 'm.0dxg6', 'm.0hgst', 'm.053kd63', 'm.056z6_', 'm.02qgqt', 'm.02655s', 'm.01llhq', 'm.026t71b', 'm.01rjtl', 'm.026db3c', 'm.06b3bx', 'm.02xcc35', 'm.01pcz9', 'm.02s8vf', 'm.06v054', 'm.01csb_', 'm.0xnc3', 'm.0pj4n', 'm.0xm_0', 'm.01ndxw', 'm.09v7n0f', 'm.0ddh63s', 'm.018n52', 'm.0626jk', 'm.0b6hgm8', 'm.04l5px', 'm.069wr', 'm.01c1ww', 'm.01_587', 'm.0235fz', 'm.03h047', 'm.031fq_', 'm.0bbbky', 'm.04kyzg', 'm.0d_sbv', 'm.029dwg', 'm.02z1r_0', 'm.016srn', 'm.01t3w_', 'm.0270lg_', 'm.0ctnsv', 'm.02ppxfg', 'm.02px9j4', 'm.01ywbz', 'm.043gpt', 'm.06_74n', 'm.02j490', 'm.01h910', 'm.0kssdg', 'm.03yj5dv', 'm.07f_x', 'm.0n6f8', 'm.065k5_', 'm.027s38x', 'm.02ykkk', 'm.01chdy', 'm.016z51', 'm.0b0s3', 'm.0163l9', 'm.06h93l', 'm.0hn9fbl', 'm.03nf04', 'm.01d_bx', 'm.07xb48', 'm.020yj1', 'm.01sp75k', 'm.03hfvl', 'm.02cdqb', 'm.0bq8p7', 'm.0h3tl82', 'm.04l20l', 'm.03wqs65', 'm.07l1h3', 'm.0g56339', 'm.01x209s', 'm.044zvm', 'm.0552zz', 'm.0d9q7w', 'm.026qc96', 'm.065zxc', 'm.016z2j', 'm.0269l3g', 'm.027_zq', 'm.0bkjs4', 'm.02l5km', 'm.019f8r', 'm.01tgq_', 'm.02djmh', 'm.0bsfy', 'm.05vslll', 'm.04kc9m', 'm.0985c0', 'm.015b67', 'm.06yks9', 'm.02_z4g', 'm.05b6sg5', 'm.04ghydx', 'm.08xh0f', 'm.05pdct', 'm.01my95', 'm.04d49y', 'm.02h8sh', 'm.0gr21', 'm.08ydvp', 'm.05k185', 'm.01zl71', 'm.074tyf', 'm.06b_0', 'm.02wxcr', 'm.053ksp', 'm.05ckg_', 'm.06c0j', 'm.0f8d6c', 'm.0lf9j', 'm.0b_lfh', 'm.0j8g6', 'm.034fn4', 'm.0264f6', 'm.02776bg', 'm.0473p_', 'm.03dlc9', 'm.02xxbs', 'm.04grr30', 'm.02pb1n', 'm.079dy', 'm.03mkf8', 'm.01jw4r', 'm.019d0l', 'm.026y49b', 'm.0bq7m', 'm.0c7wm8', 'm.05y7nf', 'm.0q9zs', 'm.023rpc', 'm.0m66w', 'm.06w6_', 'm.03wdk0', 'm.04bs3j', 'm.03z1cn', 'm.0289v04', 'm.01ls77', 'm.049kjv', 'm.0f98y', 'm.0418vd', 'm.031k24', 'm.02jj7b', 'm.051mg4', 'm.031h8r', 'm.0bjtyg', 'm.03ndwy', 'm.01qp6qt', 'm.02643n_', 'm.01rs0h', 'm.0163t3', 'm.039tcz', 'm.0lq90wy', 'm.01k5d4', 'm.077874', 'm.06dn4y', 'm.03f5mf', 'm.05579g', 'm.08m_0v', 'm.05hsn2', 'm.08c941', 'm.0cgzj', 'm.01csyt', 'm.0h96g', 'm.01ghyw', 'm.07_ty0', 'm.06zqs8n', 'm.0418ft', 'm.02m0nf', 'm.06pk8', 'm.05yvhj', 'm.05557v4', 'm.0659sj', 'm.0fpx_7', 'm.0h7h5p', 'm.07tmq9', 'm.0h_cvzm', 'm.0f3wr9', 'm.025k5p', 'm.01_j1t', 'm.01jtkg', 'm.03nk3t', 'm.04hp8s', 'm.0hqly', 'm.055m_v', 'm.03y7hp', 'm.07bkv', 'm.01p__8', 'm.06t_zq', 'm.094bfl', 'm.01xyq6', 'm.092g11', 'm.08rr56', 'm.03yx01', 'm.061wxp', 'm.013qwl', 'm.02mqc4', 'm.0crg7tp', 'm.02dc3_', 'm.033bkd', 'm.02z0ck8', 'm.01h434', 'm.01tgny', 'm.064kks2', 'm.0fml1k', 'm.05x1gn', 'm.01p1cn', 'm.04gd84', 'm.0f98y', 'm.0m3rl', 'm.0b6k9d', 'm.06608wm', 'm.04v62b', 'm.03cq1yy', 'm.07fsdg', 'm.020_qy', 'm.01cvp9', 'm.084f8h', 'm.07wbzj', 'm.01g4bk', 'm.026lr8', 'm.04p_tk', 'm.054xvs', 'm.02f_sx', 'm.05df3p', 'm.08q7kb', 'm.04z84', 'm.0gfg8x0', 'm.0fn_fh', 'm.0206mj', 'm.02yjf8', 'm.033f6x', 'm.03x2sj', 'm.0290v1', 'm.02mhfy', 'm.07h5d', 'm.02dgtb', 'm.0dk5zn', 'm.01s9ym', 'm.07vsx3', 'm.0643nx', 'm.02jxq1', 'm.01k53f', 'm.03m6v_', 'm.05z5y_', 'm.04zp_j', 'm.06dkf7', 'm.07rp8', 'm.0jdhp', 'm.01gct2', 'm.026s3n', 'm.01qm9d', 'm.01jf6n', 'm.01nr36', 'm.043p624', 'm.04y6_th', 'm.03lhmg', 'm.079vh1', 'm.04z598', 'm.07lmp', 'm.02zb92', 'm.04tmr4', 'm.0h79sh', 'm.01fnd0', 'm.02646db', 'm.024sld', 'm.0c837g', 'm.01d_c9', 'm.0q5fw', 'm.027d5g5', 'm.0g7kkb', 'm.02pqxl3', 'm.02661h', 'm.02j8fb', 'm.01l_r6', 'm.0gwyvzh', 'm.0dr5g9', 'm.06zttt', 'm.04m0nc', 'm.06t87d', 'm.0mbs_', 'm.0h1dy9t', 'm.05jyjn', 'm.01fq2k', 'm.04y0yc', 'm.03v1rk', 'm.0chdxy', 'm.04gq97', 'm.01b90h', 'm.01zbf1', 'm.03ndwy', 'm.0kn91', 'm.0pyg6', 'm.07ymn5', 'm.09nj72', 'm.07vm2b', 'm.02w09gx', 'm.022p28', 'm.01k0z1', 'm.01fxck', 'm.02fn5r', 'm.025_wg9', 'm.02ts3h', 'm.029tx1', 'm.0p720', 'm.0cm4xj', 'm.05_5txt', 'm.0gyx4', 'm.02y2nr', 'm.049dzvg', 'm.05r73g', 'm.04k5fq', 'm.02ppy7', 'm.0260x42', 'm.01l64q', 'm.013zyw', 'm.0h2ftf', 'm.01qn6k', 'm.0bgy72', 'm.030lw5', 'm.023w_z', 'm.023kzp', 'm.0166z2', 'm.05nhh1', 'm.0668g4', 'm.03lp0g', 'm.0lkgc', 'm.05zx3p', 'm.045qln', 'm.0fjgy', 'm.0dr046', 'm.047cl49', 'm.02lqby', 'm.08849', 'm.02pmfg3', 'm.07hsw6', 'm.04cxpks', 'm.07x_rh', 'm.028lkr', 'm.08887m', 'm.01bjnp', 'm.014hdb', 'm.0kcv4', 'm.02qtk76', 'm.01br1k', 'm.0gtj4r']


class FaceDataset(Dataset):
    def __init__(self, root, split, transforms=None, ignored_classes=None):
        self._root = os.path.join(root, 'images')
        if ignored_classes is None:
            ignored_classes = []

        self._class_names = [o for o in os.listdir(self._root) if os.path.isdir(os.path.join(self._root, o))]
        self._class_names = list(set(self._class_names) - set(ignored_classes))
        self._class_names.sort()

        if split == 'training':
            self._all_images, self._images_by_class = self._list_images(root, 'train.txt')
        elif split == 'validation':
            self._all_images, self._images_by_class = self._list_images(root, 'validation.txt')
        else:
            raise ValueError('Invalid split')

        self._transforms = transforms

    def _list_images(self, root, filename):
        class_indexes_by_class_name = {self._class_names[i]: i for i in range(len(self._class_names))}

        with open(os.path.join(root, filename), 'r') as image_file:
            images_lines = [line.strip() for line in image_file.readlines()]

        images = []
        images_by_class = [[] for _ in self._class_names]
        for images_line in images_lines:
            class_name, filename = images_line.split(' ')
            if class_name not in class_indexes_by_class_name:
                continue

            class_index = class_indexes_by_class_name[class_name]
            sound_index = len(images)
            images.append({
                'path': os.path.join(class_name, filename),
                'class_index': class_index
            })
            images_by_class[class_index].append({'index': sound_index})

        return images, images_by_class

    def __len__(self):
        return len(self._all_images)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self._root, self._all_images[index]['path'])).convert('RGB')
        if self._transforms is not None:
            image = self._transforms(image)

        return image, self._all_images[index]['class_index']

    def class_count(self):
        return len(self._images_by_class)

    def class_indexes(self):
        return [d['class_index'] for d in self._all_images]

    def lens_by_class(self):
        return [len(x) for x in self._images_by_class]

    def get_all_indexes(self, class_, index):
        return self._images_by_class[class_][index]['index']

    def transforms(self):
        return self._transforms


class FaceConcatDataset(Dataset):
    def __init__(self, face_datasets, transforms=None):
        if any((d.transforms() is not None for d in face_datasets)):
            raise ValueError('All face dataset ')

        self._face_datasets = face_datasets
        self._class_offsets = self._compute_class_offsets(face_datasets)

        self._transforms = transforms

    def _compute_class_offsets(self, face_datasets):
        offsets = []
        offset = 0
        for d in face_datasets:
            offsets.append(offset)
            offset += d.class_count()
        return offsets

    def __len__(self):
        return sum((len(d) for d in self._face_datasets))

    def __getitem__(self, index):
        dataset_index, index = self._transform_image_index(index)

        image, class_index = self._face_datasets[dataset_index][index]
        if self._transforms is not None:
            image = self._transforms(image)

        return image, class_index + self._class_offsets[dataset_index]

    def class_count(self):
        return sum((d.class_count() for d in self._face_datasets))

    def class_indexes(self):
        class_indexes_list = []
        for dataset_index, dataset in enumerate(self._face_datasets):
            for class_index in dataset.class_indexes():
                class_indexes_list.append(class_index + self._class_offsets[dataset_index])
        return class_indexes_list

    def lens_by_class(self):
        lens_by_class_list = []
        for d in self._face_datasets:
            lens_by_class_list += d.lens_by_class()
        return lens_by_class_list

    def get_all_indexes(self, class_index, index):
        dataset_index, class_index = self._transform_class_index(class_index)
        return self._face_datasets[dataset_index].get_all_indexes(class_index, index)

    def transforms(self):
        return self._transforms

    def _transform_image_index(self, image_index):
        for i, d in enumerate(self._face_datasets):
            if image_index < len(d):
                return i, image_index
            else:
                image_index -= len(d)

        raise IndexError(f'Image index out of range ({image_index})')

    def _transform_class_index(self, class_index):
        for i, d in enumerate(self._face_datasets):
            if class_index < d.class_count():
                return i, class_index
            else:
                class_index -= d.class_count()

        raise IndexError(f'Class index out of range ({class_index})')
