% Code used for all plots ? it seems so...

clear;
close all;
home;

% Metadata
K = 50;    % Number of arms
T = 1e5;   % K=100, T=1e5, M=70 => ~ 1 minute.    K=100, T=1e6, M=70 => ~10 minutes.
M = 70;   %1000 Number of MC samples
Nexp = 1; % Number of episodes to average the regret out
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

sR_ucb = zeros(T,1);
sR_ts = sR_ucb;
sR_wts = sR_ucb;
sR_le  = sR_ucb;
sR_f3e = sR_ucb;
sR_moss = sR_ucb;

muhat_re = zeros(K,1);
muhat_im = muhat_re;
muhat_re_moss = muhat_re;
muhat_im_moss = muhat_re;
muhat_re_f3e = muhat_re;
muhat_im_f3e = muhat_re;
muhat_re_le = muhat_re;
muhat_im_le = muhat_re;

for i = 1:Nexp
    
    ind = 1;
    p   = zeros(K,1);     % Initial weights
    p(ind) = 1;
    sp  = zeros(K,1);     % Sum of previous weights
    spw_re = zeros(K,1);  % Sum of previous weighted rewards
    spw_im = zeros(K,1);  % Sum of previous weighted rewards
    
    ind_moss = 1;
    p_moss   = zeros(K,1);     % Initial weights
    p_moss(ind_moss) = 1;
    sp_moss  = zeros(K,1);     % Sum of previous weights
    spw_re_moss = zeros(K,1);  % Sum of previous weighted rewards
    spw_im_moss = zeros(K,1);  % Sum of previous weighted rewards
    
    p_f3e   = ones(K,1)/K;    % Initial weights
    sp_f3e  = zeros(K,1);     % Sum of previous weights
    spw_re_f3e = zeros(K,1);  % Sum of previous weighted rewards
    spw_im_f3e = zeros(K,1);  % Sum of previous weighted rewards

    p_le   = ones(K,1)/K;    % Initial weights
    sp_le  = zeros(K,1);     % Sum of previous weights
    spw_re_le = zeros(K,1);  % Sum of previous weighted rewards
    spw_im_le = zeros(K,1);  % Sum of previous weighted rewards

    ind_ts = 1;
    p_ts   = zeros(K,1);     % Initial weights
    p_ts(ind_ts) = 1;
    sp_ts  = zeros(K,1);     % Sum of previous weights
    spw_re_ts = zeros(K,1);  % Sum of previous weighted rewards
    spw_im_ts = zeros(K,1);  % Sum of previous weighted rewards

    p_wts   = ones(K,1)/K;    % Initial weights
    sp_wts  = zeros(K,1);     % Sum of previous weights
    spw_re_wts = zeros(K,1);  % Sum of previous weighted rewards
    spw_im_wts = zeros(K,1);  % Sum of previous weighted rewards

    R_ucb = zeros(T,1);
    R_ts = R_ucb;
    R_wts = R_ucb;
    R_le = R_ucb;
    R_f3e = R_ucb;
    R_moss = R_ucb;
    svhat = 0;
    svhat_ts = 0;
    svhat_wts = 0;
    svhat_le = 0;
    svhat_f3e = 0;

    u = zeros(L,1);
    nexplore = 0;
    nexploit = 0;
    
    mode = 'exploration';
    factor = 1;
    explorationTime = 10;
    exploitationTime = 5;
    tt = 0;
    
        % Game
        tic
        for t = 1:T

            % UCB
                % Update empirical mean for each arm         
                for k = 1:K
                    if sp(k) == 0;
                        muhat_re(k) = Inf;
                        muhat_im(k) = Inf;
                    else
                        muhat_re(k) = spw_re(k)/sp(k);
                        muhat_im(k) = spw_im(k)/sp(k);
                    end
                end

                % Reward generation
                X_re = mu_re + randn*sigma./sqrt(2*p(ind));
                X_im = mu_im + randn*sigma./sqrt(2*p(ind));

                % Update statistics
                sp  = sp + p;
                spw_re = spw_re + p.*X_re;
                spw_im = spw_im + p.*X_im;
                for k = 1:K
                    svhat = svhat + (sqrt(2*p(k))*(X_re(k) - m_re_post(k)))^2;
                    svhat = svhat + (sqrt(2*p(k))*(X_im(k) - m_im_post(k)))^2;
                end
                vhat = svhat/(2*t);

                % Expected Cumulative Regret computation
                R_ucb(t) = R_ucb(max(t-1,1)) + max(0,maxmu^2 - p'*(abs(G).^2));

                % Udapte p    
                b = muhat_re.^2 + muhat_im.^2 + sqrt(.5*sigma^2*log(t+1)./sp);
                [~,ind] = max(b);
                p = zeros(K,1);
                p(ind) = 1;
                
            % MOSS
                % Update posterior for each arm
                for k = 1:K
                    if sp_moss(k) == 0;
                        muhat_re_moss(k) = Inf;
                        muhat_im_moss(k) = Inf;
                    else
                        muhat_re_moss(k) = spw_re_moss(k)/sp_moss(k);
                        muhat_im_moss(k) = spw_im_moss(k)/sp_moss(k);
                    end
                end

                % Reward generation
                X_re = mu_re + randn*sigma./sqrt(2*p_moss(ind_moss));
                X_im = mu_im + randn*sigma./sqrt(2*p_moss(ind_moss));

                % Update statistics
                sp_moss  = sp_moss + p_moss;
                spw_re_moss = spw_re_moss + p_moss.*X_re;
                spw_im_moss = spw_im_moss + p_moss.*X_im;

                % Expected Cumulative Regret computation
                R_moss(t) = R_moss(max(t-1,1)) + max(0,maxmu^2 - p_moss'*(abs(G).^2));

                % Udapte p    
                
                b_moss = Inf*ones(K,1);
                for k = 1:K
                    if ~isnan(sp_moss(k))
                        b_moss(k) = muhat_re_moss(k)^2 + muhat_im_moss(k)^2 + sqrt(max(log(t/(K*sp_moss(k))), 0)'./sp_moss(k));
                    end
                end
                [~,ind_moss] = max(b_moss);
                p_moss = zeros(K,1);
                p_moss(ind_moss) = 1;
            
            % Fixed Exploration - Exponential exploitation (F3E)
                % Update posterior for each arm
                for k = 1:K
                    if sp_f3e(k) == 0;
                        muhat_re_f3e(k) = Inf;
                        muhat_im_f3e(k) = Inf;
                    else
                        muhat_re_f3e(k) = spw_re_f3e(k)/sp_f3e(k);
                        muhat_im_f3e(k) = spw_im_f3e(k)/sp_f3e(k);
                    end
                end

                % Reward generation
                X_re = mu_re + randn*sigma./sqrt(2*p_f3e);
                X_im = mu_im + randn*sigma./sqrt(2*p_f3e);

                % Update statistics
                sp_f3e  = sp_f3e + p_f3e;
                spw_re_f3e = spw_re_f3e + p_f3e.*X_re;
                spw_im_f3e = spw_im_f3e + p_f3e.*X_im;
                for k = 1:K
                    svhat_f3e = svhat_f3e + (sqrt(2*p_f3e(k))*(X_re(k) - muhat_re_f3e(k)))^2;
                    svhat_f3e = svhat_f3e + (sqrt(2*p_f3e(k))*(X_im(k) - muhat_im_f3e(k)))^2;
                end
                vhat_f3e = svhat_f3e/(2*t);

                % Expected Cumulative Regret computation
                R_f3e(t) = R_f3e(max(t-1,1)) + max(0,maxmu^2 - p_f3e'*(abs(G).^2));

                % Udapte p    
                b_f3e = muhat_re_f3e.^2 + muhat_im_f3e.^2 + sqrt(3*sigma^2*log(t+1)./sp_f3e);
                if strcmp(mode,'exploration') > 0
                    p_f3e = b_f3e; % + 7*log(t+1)./sp(k);
                    p_f3e = p_f3e/sum(p_f3e);
                    if tt == explorationTime
                        tt = 0;
                        mode = 'exploitation';
                    end
                else
                    p_f3e = ones(K,1)*eps;
                    [~,ind_f3e] = max(b);
                    p_f3e(ind_f3e) = 1;
                    if tt == exploitationTime*factor
                        tt = 0;
                        mode = 'exploration';
                        factor = factor*2;
                    end
                end
                tt = tt + 1;
                
            % Leader-based
                % Update posterior for each arm
                for k = 1:K
                    if sp_le(k) == 0;
                        muhat_re_le(k) = Inf;
                        muhat_im_le(k) = Inf;
                    else
                        muhat_re_le(k) = spw_re_le(k)/sp_le(k);
                        muhat_im_le(k) = spw_im_le(k)/sp_le(k);
                    end
                end

                % Reward generation
                X_re = mu_re + randn*sigma./sqrt(2*p_le);
                X_im = mu_im + randn*sigma./sqrt(2*p_le);

                % Update statistics
                sp_le  = sp_le + p_le;
                spw_re_le = spw_re_le + p_le.*X_re;
                spw_im_le = spw_im_le + p_le.*X_im;
                
                % Variance estimation
                for k = 1:K
                    svhat_le = svhat_le + (sqrt(2*p_le(k))*(X_re(k) - muhat_re_le(k)))^2;
                    svhat_le = svhat_le + (sqrt(2*p_le(k))*(X_im(k) - muhat_im_le(k)))^2;
                end
                vhat_le = svhat_le/(2*t);

                % Expected Cumulative Regret computation
                R_le(t) = R_le(max(t-1,1)) + max(0,maxmu^2 - p_le'*(abs(G).^2));

                % Udapte p    
                b_le = muhat_re.^2 + muhat_im.^2 + sqrt(0.01*sigma^2*log(t+1)./sp_le);
                [maxmean,ind_maxmean] = max(muhat_re_le.^2 + muhat_im_le.^2);
                bb = b_le;
                bb(ind_maxmean) = 0;
                if sum(maxmean < bb) > 0
                    p_le = b_le;
                    p_le = p_le/sum(p_le);
                    for k = 1:K
                        if p_le(k) < eps
                            p_le(k) = eps;
                        end
                    end
                    nexplore = nexplore + 1;
                else
                    p_le = eps*ones(K,1);
                    p_le(ind_maxmean) = 1;
                    nexploit = nexploit + 1;
                end

            % TS
                m_re_post_ts = lambda^2*spw_re_ts ./ (sigma^2 + lambda^2*sp_ts); % Re( post_mean )
                m_im_post_ts = lambda^2*spw_im_ts ./ (sigma^2 + lambda^2*sp_ts); % Im( post_mean )
                v_post_ts = lambda^2 ./ (1 + lambda^2*sp_ts/sigma^2);   % Posterior variance of all arms

                % Reward generation
                X_re = mu_re + randn*sigma./sqrt(2*p_ts(ind_ts));
                X_im = mu_im + randn*sigma./sqrt(2*p_ts(ind_ts));

                % Update statistics
                sp_ts  = sp_ts + p_ts;
                spw_re_ts = spw_re_ts + p_ts.*X_re;
                spw_im_ts = spw_im_ts + p_ts.*X_im;
                for k = 1:K
                    svhat_ts = svhat_ts + (sqrt(2*p_ts(k))*(X_re(k) - m_re_post_ts(k)))^2;
                    svhat_ts = svhat_ts + (sqrt(2*p_ts(k))*(X_im(k) - m_im_post_ts(k)))^2;
                end
                vhat_ts = svhat_ts/(2*t);

                % Expected Cumulative Regret computation
                R_ts(t) = R_ts(max(t-1,1)) + max(0,maxmu^2 - p_ts'*(abs(G).^2));

                % Udapte p    
                samples_re_ts = diag(sqrt(v_post_ts))*randn(K,1) + m_re_post_ts;
                samples_im_ts = diag(sqrt(v_post_ts))*randn(K,1) + m_im_post_ts;
                samples_ts = samples_re_ts.^2 + samples_im_ts.^2;
                [~,ind_ts] = max(samples_ts);
                p_ts = zeros(K,1);
                p_ts(ind_ts) = 1;

            % WTS
                m_re_post_wts = lambda^2*spw_re_wts ./ (sigma^2 + lambda^2*sp_wts); % Re( post_mean )
                m_im_post_wts = lambda^2*spw_im_wts ./ (sigma^2 + lambda^2*sp_wts); % Im( post_mean )
                v_post_wts = lambda^2 ./ (1 + lambda^2*sp_wts/sigma^2);   % Posterior variance of all arms

                % Reward generation
                X_re = mu_re + randn*sigma./sqrt(2*p_wts);
                X_im = mu_im + randn*sigma./sqrt(2*p_wts);

                % Update statistics
                sp_wts  = sp_wts + p_wts;
                spw_re_wts = spw_re_wts + p_wts.*X_re;
                spw_im_wts = spw_im_wts + p_wts.*X_im;
                for k = 1:K
                    svhat_wts = svhat_wts + (sqrt(2*p_wts(k))*(X_re(k) - m_re_post_wts(k)))^2;
                    svhat_wts = svhat_wts + (sqrt(2*p_wts(k))*(X_im(k) - m_im_post_wts(k)))^2;
                end
                vhat_wts = svhat_wts/(2*t);

                % Expected Cumulative Regret computation
                R_wts(t) = R_wts(max(t-1,1)) + max(0,maxmu^2 - p_wts'*(abs(G).^2));

                % Udapte p    
                samples_re_wts = diag(sqrt(v_post_wts))*randn(K,M) + m_re_post*ones(1,M);
                samples_im_wts = diag(sqrt(v_post_wts))*randn(K,M) + m_im_post_wts*ones(1,M);
                samples_wts = samples_re_wts.^2 + samples_im_wts.^2;
                max_arm = zeros(K,1);
                for j = 1:M
                [~, indmax] = max(samples_wts(:,j));
                max_arm(indmax) = max_arm(indmax) + 1;
                end
                p_wts = max_arm/M + eps;
                p_wts = p_wts/sum(p_wts);          


                %  Online power allocation
    %              rplot2(K,Gplot, p_wts, index_maxmu, t);
        end
    
    % Regret accumulation
    sR_ucb = sR_ucb + R_ucb; 
    sR_ts   = sR_ts   + R_ts;
    sR_wts  = sR_wts  + R_wts;
    sR_le   = sR_le   + R_le;
    sR_f3e  = sR_f3e  + R_f3e;
    sR_moss  = sR_moss+ R_moss;
end
toc

%% SAVE
save('results.mat');

R_ucb = sR_ucb/Nexp;
R_ts = sR_ts/Nexp;
R_wts = sR_wts/Nexp;
R_le = sR_le/Nexp;
R_f3e = sR_f3e/Nexp;
R_moss = sR_moss/Nexp;
figure
plot(R_ucb);
hold on
plot(R_ts);
plot(R_wts);
plot(R_le);
plot(R_f3e);
plot(R_moss)
legend('UCB-V', 'TS', 'WTS', 'LB','F3E', 'MOSS');


