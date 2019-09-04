[x,t] = glass_dataset;
net = feedforwardnet(5);
view(net)
[net,tr] = train(net,x,t);
plotperform(tr)

%%

round(net([[1.51711000000000;14.2300000000000;0;2.08000000000000;73.3600000000000;0;8.62000000000000;1.67000000000000;0]]))