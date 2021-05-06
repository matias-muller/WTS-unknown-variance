clear;
close all;
home;
clc;

n = 1e2;
nexp = 1e4;
mu = [.2; -.7];
sigma2 = 0.4;
sigma = sqrt(sigma2);
p = rand(1,n);
var = zeros(2);
for k = 1:nexp
    X = zeros(2,n);
    barx = zeros(2,1);
    for i = 1:n
        X(:,i) = mu + sigma/sqrt(p(i))*randn(2,1);
        barx = barx + p(i)*X(:,i);
    end
    barx = barx/sum(p);
    var = var + (barx - mu)*(barx - mu)';
end
var = var/nexp;

vart = n/((sum(p))^2)*sigma2/2*eye(2);