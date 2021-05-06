clear all
close all
home
clc


T = 500;
N = 100000;

sigma = 1;

first = zeros(T,1);
second = first;
third = first;
fourth = first;

f = 1./((1:T))';
g = sigma^2./((1:T)');

for t = 1:T
    x = f(t) + sqrt(g(t)).*randn(N,1); % N samples of x~(1/t,sigma^2)
    first(t) = sum(x)/N;
    second(t) = sum(x.^2)/N;
    third(t) = sum(x.^3)/N;
    fourth(t) = sum(x.^4)/N;
end

figure
plot(first);


figure
plot(second);
hold on
plot(sigma^2*ones(T,1) + 1./((1:T).^2)');

figure
plot(third);
hold on
plot(1./((1:T).^3') +3*sigma^2./((1:T)'));

figure
plot(fourth);
hold on
plot(3*sigma^4 + 1./((1:T).^4)' + 6*sigma^2./((1:T).^2)');

