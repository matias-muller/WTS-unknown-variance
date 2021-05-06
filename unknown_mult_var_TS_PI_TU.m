
clear;
close all;
home;

% Metadata
K = 4;                  % Number of arms
T = 1e3; %1e4;                %
Nexp = 20;               % Number of episodes to average the regret out
%sigma2 = rand(K,1);     % Noise variances
%sigma = sqrt(sigma2);   % Standard deviation of noise
bH = [-2 1 -8 2];
aH = [1 0 0 0];
H = tf(bH,aH,1);
sigma = zeros(K,1);
for k = 1:K
    sigma(k) = abs(evalfr(H,1i*2*pi*k/(2*K+1)));
end
sigma2 = sigma.^2;

factor = max(1,T/1e3);             % downsampling factor
L = 100;                % length of power-iterations signals.

% Filter: Several maxima.
load('Num.mat');        % Filter coefficients (via fdatool)
a = [1 zeros(1,length(Num)-1)];
b = Num;
% Filter gains
Gtf = tf(b,a,1);
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
sR2 = sR;
n = 4;
p = zeros(K,1);
mse_pi = zeros(T,1);
mse_pi_aux = mse_pi;
mse_ts_aux = mse_pi;
mse_ts = mse_pi;
uu = rand(L,1);

tic
for i = 1:Nexp 
    R = zeros(T,1);
    Nk = zeros(K,1);
    sx_re = zeros(K,1);
    sx_im = sx_re;
    sS = zeros(K,1);
    S = sS;
    tt = 1;
    for nn = 1:n
        for k = 1:K
            arm = k;
            X_re = mu_re(arm) + randn*sigma(arm)/sqrt(2);
            X_im = mu_im(arm) + randn*sigma(arm)/sqrt(2);
            Nk(arm) = Nk(arm) + 1;
            sx_re(arm) = sx_re(arm) + X_re;
            sx_im(arm) = sx_im(arm) + X_im;
            xx_re = sx_re./Nk;
            xx_im = sx_im./Nk;
            sS(arm) = sS(arm) + X_re^2 + X_im^2;
            S(arm) = sS(arm) - Nk(arm)*(xx_re(arm)^2+xx_im(arm)^2);
            R(tt) = R(max(tt-1,1)) + max(0,maxmu^2 - (abs(G(arm)).^2));
            tt = tt + 1;
        end
    end
    samples_re = zeros(K,1);
    samples_im = samples_re;
    r = samples_re;
    for k = 1:K
        % 1. Generate a random radius around xx
        u = rand;
        %r(k) = sqrt(  S(k)/(2*Nk(k))*((n-3)/(n-2)*u^(1/(-Nk(k)+2))-1)  );
        % Shouldn't n's be Nk(k) ????
         r(k) = sqrt(    S(k)/Nk(k)*(  (u)^(1/(2-Nk(k)))- 1  )    );
        % 2. For a fixed radius, take a uniform sample around the circle
        % parametrizing by theta => (x,y) = (r*cos(theta), r*sin(theta)),
        % with theta = 2*pi*rand();
        theta = 2*pi*rand;
        samples_re(k) = r(k)*cos(theta) + xx_re(k);
        samples_im(k) = r(k)*sin(theta) + xx_im(k);
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
        xx_re = sx_re./Nk;
        xx_im = sx_im./Nk;
        sS(arm) = sS(arm) + X_re^2 + X_im^2;
        S(arm) = sS(arm) - Nk(arm)*(xx_re(arm)^2+xx_im(arm)^2);
        % Expected Cumulative Regret computation
        R(t) = R(max(t-1,1)) + max(0,maxmu^2 - p'*(abs(G).^2));

        % Udapte p
        samples_re = zeros(K,1);
        samples_im = samples_re;
        r = samples_re;
        for k = 1:K
            % 1. Generate a random radius around xx
            u = rand;
            %r(k) = sqrt(  S(k)/(2*Nk(k))*(u^(1/(-Nk(k)+2))-1)  );
            % shouldn't n's be Nk(k) ????
            r(k) = sqrt(    S(k)/Nk(k)*(  (u)^(1/(2-Nk(k)))- 1  )    );
            % 2. For a fixed radius, take a uniform sample around the circle
            % parametrizing by theta => (x,y) = (r*cos(theta), r*sin(theta)),
            % with theta = 2*pi*rand();
            theta = 2*pi*rand;
            samples_re(k) = r(k)*cos(theta) + xx_re(k);
            samples_im(k) = r(k)*sin(theta) + xx_im(k);
        end
        samples = samples_re.^2 + samples_im.^2;
        [~,arm] = max(samples);
        p = zeros(K,1);
        p(arm) = 1;
        
        [~,most_played_arm] = max(Nk);
        beta_ts = sqrt( xx_im(most_played_arm)^2 + xx_re(most_played_arm)^2 );
        mse_ts_aux(t) = (beta - beta_ts)^2;
        

        %  Online power allocation
%                       rplot2(K,Gplot, p, index_maxmu, t);
%         errorbar(xx_re.^2+xx_im.^2,S./Nk);
%         hold on
%         errorbar(xx_re.^2+xx_im.^2,sigma2);
%         ylim([0 2]);
%         hold off;
%         pause(0.1)
    end
    
    % Regret accumulation
    sR = sR + R;
    sR2 = sR2 + R.^2;
    
    % Beta TS
    mse_ts = mse_ts + mse_ts_aux;
    
    % Power iterations
    for t = 1:T
        y = filter(b,a,uu) + filter(bH,aH,randn(L,1));
        ytilde = flip(y);
        mu = norm(ytilde,2)/sqrt(L);
        uu = ytilde/mu;
        beta_pi = uu'*ytilde/L;
        mse_pi_aux(t) = (beta - beta_pi)^2;
    end
    
    mse_pi = mse_pi + mse_pi_aux;
    
end
toc

%% SAVE
save('results.mat');

sR = sR/Nexp;
% varR = (sR2 - Nexp*sR.^2)/Nexp;


figure
% plot(sR);
hold on

legend('TS', '\Delta(log(1+\Delta/\sigma^2))^{-1}');
set(gca, 'XScale', 'log')


%%
% Fit FIR filter
LL = 20;
Phi = zeros(L, LL);
y = filter(b,a,uu) + filter(bH,aH,randn(L,1));
for i = LL+1:L
    Phi(i,:) = uu(i-LL:i-1)';
end
Phi = Phi(LL+1:end,:);

theta = Phi\y(LL+1:end);
theta = flip(theta);
Ghat = tf(theta',[1,zeros(1,LL-1)],1);
beta_TU = norm(Ghat,'inf');

%%


mse_ts = mse_ts/Nexp;
mse_pi = mse_pi/Nexp;
figure
plot(mse_ts);
hold on
plot(mse_pi);
legend('ts','pi');






