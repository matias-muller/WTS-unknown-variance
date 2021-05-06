clear;
close all;
home;

% Metadata
K = 100;    % Number of arms
T = 1e5;   % Number of iterations % 20 000 per night, under 1e5 MC points. N=1000,M=1000 ~ 20sec.
M = 70;   %1000 Number of MC samples
Nexp = 1; % Number of experiments to compute a variance estimator
sigma = .1;  % Standard deviation of noise
lambda = 1; % Prior standard deviation of mean rewards
L = 100;    % length of the input for power iterations
eps = 1e-15; % regulations term for p
tolerance = 1e-7;   % treshold to consider a frequency in the Hinf-norm est.
m_re_post = zeros(K,1);
m_im_post = m_re_post;
Ghat = 0;
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
G = G(2:K+1);
absG = abs(G);
Nplot = 1000;
Gplot = abs(freqz(b,a,Nplot));

% True mean reward distribution
mu_re = real(G);
mu_im = imag(G);
[maxmu, index_maxmu] = max(abs(G));
beta = maxmu;

sR = zeros(T,1);

for i = 1:Nexp
    p   = ones(K,1)/K;    % Initial weights
    sp  = zeros(K,1);     % Sum of previous weights
    spw_re = zeros(K,1);  % Sum of previous weighted rewards
    spw_im = zeros(K,1);  % Sum of previous weighted rewards

    p_ts   = ones(K,1)/K;    % Initial weights
    sp_ts  = zeros(K,1);     % Sum of previous weights
    spw_re_ts = zeros(K,1);  % Sum of previous weighted rewards
    spw_im_ts = zeros(K,1);  % Sum of previous weighted rewards
    spw_abs = zeros(K,1);    % Sum of previous w. quadratic rewards

    R = zeros(T,1);
    svhat = 0;

    u = zeros(L,1);
    mode = 'exploration';
    factor = 1;
    explorationTime = 1000;
    exploitationTime = 5;
    tt = 0;
    
    nexplore = 0;
    nexploit = 0;
    % Game
    tic
    for t = 1:T
        % Update posterior for each arm
        m_re_post = lambda^2*spw_re ./ (sigma^2 + lambda^2*sp); % Re( post_mean )
        m_im_post = lambda^2*spw_im ./ (sigma^2 + lambda^2*sp); % Im( post_mean )

        % Reward generation
        X_re = mu_re + randn(K,1)*sigma./sqrt(2*p);
        X_im = mu_im + randn(K,1)*sigma./sqrt(2*p);
        X = X_re + J*X_im;

        % Update statistics
        sp  = sp + p;
        spw_re = spw_re + p.*X_re;
        spw_im = spw_im + p.*X_im;
        for k = 1:K
            svhat = svhat + (sqrt(2*p(k))*(X_re(k) - m_re_post(k)))^2;
            svhat = svhat + (sqrt(2*p(k))*(X_im(k) - m_im_post(k)))^2;
        end
        vhat = svhat/(2*K*t);

        % Expected Cumulative Regret computation
        R(t) = R(max(t-1,1)) + maxmu^2 - sum(p'*(abs(G).^2));

        % Udapte p    
        b = m_re_post.^2 + m_im_post.^2 + sqrt(3/2*sigma^2*log(t+1)./sp);
        [maxmean,ind_maxmean] = max(m_re_post.^2 + m_im_post.^2);
        bb = b;
        bb(ind_maxmean) = 0;
        if sum(maxmean < bb) > 0
            p = b;
            p = p/sum(p);
            nexplore = nexplore + 1;
            for k = 1:K
                if p(k) < eps
                    p(k) = eps;
                end
            end
        else
            p = eps*ones(K,1);
            p(ind_maxmean) = 1;
            nexploit = nexploit + 1;
        end

        %   Comment this to get the result faster.
%          rplot2(K,Gplot, p, index_maxmu, t);
    end
    sR = sR + R; 
end
toc

%% SAVE
save('results.mat');

R = sR/T;
figure
plot(R);


