# import torch
#
# ss = torch.ones([2, 2, 3, 4])
# for i in range(2):
#     for j in range(3):
#         ss[i][0][j][3] = 0
#         ss[i][1][j][3] = 0
# print(ss)
#
# tt = torch.zeros([2, 1, 3, 4])
# for i in range(2):
#     for j in range(3):
#         tt[i][0][j][3] = -10000
#         if j == 2:
#             tt[i][0][j][3] = -20000
#         if i == 1:
#             tt[i][0][2][3] = -30000
# print(tt)
# kk = ss + tt
# print(kk)

print([0] * 8)