% Code used for all plots ? it seems so...

clear;
close all;
home;

% Metadata
K = 100;    % Number of arms
N = 1e5;   % Time horizon
Nexp = 1; % Number of experiments to compute a variance estimator
sigma = .1;  % Standard deviation of noise
lambda = 1; % Prior standard deviation of mean rewards
L = 2*K+1;    % length of the input for power iterations

J = sqrt(-1); 

% Filter properties
% Filter 1: One maximum.
r = 0.95;
w0 = 75/100*pi;
a = [1 -2*r*cos(w0) r^2];
b = 1-r;
% Filter 2: Several maxima.
load('Num.mat');        % Filter coefficients (via fdatool)
a = [1 zeros(1,length(Num)-1)];
b = Num;
% Filter gains
G = freqz(b,a,K+1);
Nplot = 1000;
[aGplot,w] = freqz(b,a,Nplot);
Gplot = abs(aGplot);

% True mean reward distribution
G = G(2:end);    % drop G at w = 0
mu_re = real(G);
mu_im = imag(G);
[maxmu, index_maxmu] = max(abs(G));
beta = maxmu;

u = randn(L,1);
u = u/norm(u);
p = zeros(N,K);
X = zeros(N,K);
R = zeros(N,1);

% Game
tic
figure
for i = 1:N  
    % Perform an experiment
    y = filter(b,a,u) + sigma*randn(L,1);
    Y = fft(y);
    Y = Y(2:K+1);
    
    % Corresponding p
    U = fft(u);
    U = U(2:K+1);
    p(i,:) = abs(U).^2;
    p(i,:) = p(i,:)/sum(p(i,:));
    
    % Reward generation
    X(i,:) = Y./U;
    
    % Non-expected cumulative Regret computation
    R(i) = R(max(i-1,1)) + maxmu^2 - sum(p(i,:)*(abs(G).^2));
    
    % Power iterations (update u)
    ytilde = flip(y);
    mu = norm(ytilde,2)/sqrt(L);
    u = ytilde/mu;
    %[~, khat_index] = max(p(i,:));
    %u = u/i + (1-1/exp(i*i))*sin(2*pi/L*(khat_index-1)*(1:L)');
    U = fft(u);
    absU = abs(U)/sum(abs(U));
    phaseU = phase(U);
    eta = 1/sqrt(i);
    for k = 2:K+1
       absU(k) = exp(eta*sum(p(1:i,k-1).*(abs(X(1:i,k-1)).^2)));
       absU(2*K+3-k) = exp(eta*sum(p(1:i,k-1).*(abs(X(1:i,k-1)).^2)));
    end
    U = absU.*exp(J*phaseU);
    u = real(ifft(U));
    u = u/norm(u)*14.1774;
    %beta_pi = u'*ytilde/L;
    
%   Comment this to get the result faster.
%     if sum(i==(50500:52000) > 0)
%         rplot(K,Gplot, p, index_maxmu, i);
%     end
end
toc

%% plot PI regret

figure
plot(R);
title('PI regret');
