
clear;
close all;
home;

% Metadata
K = 2;                  % Number of arms
T = 1e3;                % K=100, T=1e5, M=70 => ~ 1 minute.    K=100, T=1e6, M=70 => ~10 minutes.
Nexp = 30;               % Number of episodes to average the regret out
sigma2 = rand(K,1);     % Noise variances
sigma = sqrt(sigma2);   % Standard deviation of noise
lambda = 1;             % Prior standard deviation of mean rewards
eps = 1e-30;            % regulations term for p
tolerance = 1e-7;       % treshold to consider a frequency in the Hinf-norm est.
m_re_post = zeros(K,1);
m_im_post = m_re_post;
Ghat = 0;
J = sqrt(-1); 


% Filter: Several maxima.
load('Num.mat');        % Filter coefficients (via fdatool)
a = [1 zeros(1,length(Num)-1)];
b = Num;
% Filter gains
G = freqz(b,a,K+1);
G = G(2:K+1);
absG = abs(G);
Nplot = 1000;
Gplot = abs(freqz(b,a,Nplot));
a = -1;

% True mean reward distribution
mu_re = real(G);
mu_im = imag(G);
[maxmu, index_maxmu] = max(abs(G));
beta = maxmu;

sR = zeros(T,1);
n = max(2,3-floor(2*a));
p = zeros(K,1);

tic
for i = 1:Nexp 
    R = zeros(T,1);
    Nk = zeros(K,1);
    sx_re = zeros(K,1);
    sx_im = sx_re;
    S_re = zeros(K,1);
    S_im = S_re;
    tt = 1;
    for nn = 1:n
        for k = 1:K
            arm = k;
            X_re = mu_re(arm) + randn*sigma(arm)/sqrt(2);
            X_im = mu_im(arm) + randn*sigma(arm)/sqrt(2);
            Nk(arm) = Nk(arm) + 1;
            sx_re(arm) = sx_re(arm) + X_re;
            sx_im(arm) = sx_im(arm) + X_im;
            x_re = sx_re./Nk;
            x_im = sx_im./Nk;
            S_re(arm) = S_re(arm) + (X_re - x_re(arm))^2;
            S_im(arm) = S_im(arm) + (X_im - x_im(arm))^2;
            R(tt) = R(max(tt-1,1)) + max(0,maxmu^2 - (abs(G(arm)).^2));
            tt = tt + 1;
        end
    end
    samples_re = zeros(K,1);
    samples_im = samples_re;
    for k = 1:K
        samples_re(k) = trnd(Nk(k)+2*a-1)*sqrt(S_re(k)/(Nk(k)*(Nk(k)+2*a-1))) + x_re(k);
        samples_im(k) = trnd(Nk(k)+2*a-1)*sqrt(S_im(k)/(Nk(k)*(Nk(k)+2*a-1))) + x_im(k);
    end
    samples = samples_re.^2 + samples_im.^2;
    [~,arm] = max(samples);
    p = zeros(K,1);
    p(arm) = 1;
    
    
    for t = n*K+1:T
        % Reward generation
        X_re = mu_re(arm) + randn*sigma(arm)/sqrt(2);
        X_im = mu_im(arm) + randn*sigma(arm)/sqrt(2);

        % Update statistics
        Nk(arm) = Nk(arm) + 1;
        sx_re(arm) = sx_re(arm) + X_re;
        sx_im(arm) = sx_im(arm) + X_im;
        x_re = sx_re./Nk;
        x_im = sx_im./Nk;
        S_re(arm) = S_re(arm) + (X_re - x_re(arm))^2;
        S_im(arm) = S_im(arm) + (X_im - x_im(arm))^2;
        
        % Expected Cumulative Regret computation
        R(t) = R(max(t-1,1)) + max(0,maxmu^2 - p'*(abs(G).^2));

        % Udapte p
        samples_re = zeros(K,1);
        samples_im = samples_re;
        for k = 1:K
            samples_re(k) = trnd(Nk(k)+2*a-1)*sqrt(S_re(k)/(Nk(k)*(Nk(k)+2*a-1))) + x_re(k);
            samples_im(k) = trnd(Nk(k)+2*a-1)*sqrt(S_im(k)/(Nk(k)*(Nk(k)+2*a-1))) + x_im(k);
        end
        samples = samples_re.^2 + samples_im.^2;
        [~,arm] = max(samples);
        p = zeros(K,1);
        p(arm) = 1;
        

        %  Online power allocation
        %              rplot2(K,Gplot, p, index_maxmu, t);
    end
    
    % Regret accumulation
    sR = sR + R;
    
    
    
end
toc

%% SAVE
save('results.mat');

sR = sR/Nexp;
figure
plot(sR+ones(T,1));
hold on

% Single freq lower bound
single_constant = 0;
for k = 1:K
    if k ~= index_maxmu
        single_constant = single_constant + sigma2(k)*(maxmu^2-abs(G(k))^2)/(abs(G(index_maxmu)-G(k))^2);
    end
end

time = 1:T;
plot(time,single_constant*log(time),'--')


    % Single freq lower bound
mult_constant = 0;
for k = 1:K
    if k ~= index_maxmu
        mult_constant = mult_constant + (maxmu^2-abs(G(k))^2)/(log(1 + abs(G(index_maxmu)-G(k))^2/sigma2(k)));
    end
end
plot(time,mult_constant*log(time),'g')


legend('TS', '\sigma^2/\Delta', '\Delta(log(1+\Delta/\sigma^2))^{-1}');
set(gca, 'XScale', 'log')

