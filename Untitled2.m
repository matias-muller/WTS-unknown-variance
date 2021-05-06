clear
clc
close all

N = 1e5;
K = 3;
mu = [5.4   5     4.5;...
      2.1   4.2   1.3];
sigma = [1000 1000 1000];
count = zeros(1,K);

for i = 1:N
    p = rand(K,1);
    p = p/sum(p);
    p = [1 1 1];
    samples = zeros(K,1);
    for j = 1:K
        samples(j) = norm(mu(:,j) + diag(sqrt(sigma(j))/p(j))*randn(2,1));
    end
    [~,k] = max(samples);
    count(k) = count(k) + 1;
end
    
figure
stem(count/N);
xlim([0, K+1]);

disp(count/N)