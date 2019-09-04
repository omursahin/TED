

%%
net = fitnet(18,'trainbr');


giris = [1 1;1 2;2 2;2 3;3 3; 3 4;5 5;7 8];
cikis = [1 2 4 6 9 12 25 56];

[net,tr] = train(net,giris',cikis);
%%
round(net([2;7]))
