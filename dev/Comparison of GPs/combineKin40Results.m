s_rmse = [0.248184709
0.243750303
0.233733355
0.233453416
0.232866966
0.204696685
0.304448793
0.264796049
];
s_time = [208.2599231
323.3306291
433.9118597
7.12E+02
5.45E+02
1.93E+03
113.9010001
392.9861731
];
s_num_eval = [200
200
200
200
100
100
200
200
];
s_n_train = [10000
10000
10000
10000
10000
5000
2000
2000
];
s_n_sparse = [300
400
500
700
900
2500
500
1000
];

d_rmse = [0.145807088
0.162849433
0.176559732
0.494973198
0.398909536
0.259027565
0.160674336
0.185349082
];

d_time = [2.05E+02
1.36E+02
94.9843449
51.0714749
12.55079768
27.5982782
147.9098583
97.26649232
];

d_num_eval = [100
100
100
100
100
100
100
100
];

d_n_train = [10000
10000
10000
10000
10000
10000
10000
10000
];

d_M = [3
4
5
256
64
16
4
6
];

f_rmse = [0.150587544
0.210499657
0.201627907
0.182580325
0.166030573
0.16643129
0.271310324
0.226057014
];

f_time = [1.92E+02
33.6959956
53.26297304
104.5342936
125.0329589
1.49E+02
9.552864744
21.51434917
];

f_num_eval = [100
100
100
100
100
100
100
100
];

f_n_train = [5000
2500
3000
3500
4000
4500
1500
2000
];


sgp_history = combine(s_rmse, s_time, s_num_eval, s_n_train, 'sparse', s_n_sparse);
dgp_history = combine(d_rmse, d_time, d_num_eval, d_n_train, 'dist', d_M);
fgp_history = combine(f_rmse, f_time, f_num_eval, f_n_train, 'full');

save('combined_Kin40k_results.mat', 'fgp_history', 'sgp_history', 'dgp_history');


function history = combine(rmse, time, num_eval, n_train, type, m)
for i=1:length(rmse)
    history(i).rmse = rmse(i);
    history(i).time = time(i);
    history(i).params.num_eval = num_eval(i);
    history(i).params.n_train = n_train(i);
    if strcmp(type, 'sparse')
        history(i).params.n_sparse = m(i);
    elseif strcmp(type, 'dist')
        history(i).params.M = m(i);
    end
end
end

