
clear;
close all;
home;

% Metadata
K = 4;                  % Number of arms
T = 1e4;                %
Nexp = 50;               % Number of episodes to average the regret out
% sigma2 = rand(K,1);     % Noise variances
% sigma = sqrt(sigma2);   % Standard deviation of noise
factor = max(1,T/1e3);             % downsampling factor
L = 1000;
Nplot = 100;

% Noise filter
bH = .025*[-2 1 -8 2]; %[-2 1 -8 2]; old command before last minute change
aH = [1 0 0 0];
% H = tf(bH,aH,1);
H = freqz(bH,aH,K+1);
H = H(2:K+1);
factorH = 0.025; %max(abs(H))*.5;
H = H*factorH; %H/factorH; old command before last minute change
sigma = zeros(K,1);
for k = 1:K
%     sigma(k) = abs(evalfr(H,1i*2*pi*k/(2*K+1)));
    sigma = abs(H);
end
sigma2 = sigma.^2;
Hplot = abs(freqz(bH,aH,Nplot));
Hplot = Hplot*factorH; %Hplot/factorH; old command before last minute change

% Filter: Several maxima.
load('Num.mat');        % Filter coefficients (via fdatool)
a = [1 zeros(1,length(Num)-1)];
b = Num;
% Filter gains
G = freqz(b,a,K+1);
G = G(2:K+1);
absG = abs(G);

Gplot = abs(freqz(b,a,Nplot));



% True mean reward distribution
mu_re = real(G);
mu_im = imag(G);
[maxmu, index_maxmu] = max(abs(G));
beta = maxmu;

beta_ts = zeros(T,1);   % betahat = norm(largest_mu)
beta_ts_arm = beta_ts;  % betahat = norm(mu(most_played_arm))
beta_pi = beta_ts;
beta_tu = 0;
beta_tu2 = 0;
mse_ts = beta_ts;
mse_ts_arm = beta_ts;
mse_pi = beta_ts;
mse_tu = 0;
mse_tu2 = 0;

sR = zeros(T,1);
sR2 = sR;
n = 4;
p = zeros(K,1);

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
            beta_ts(tt) = max(abs(xx_re + 1i*xx_im));
            [~,most_played_arm] = max(Nk);
            beta_ts_arm(tt) = abs(xx_re(most_played_arm) + 1i*xx_im(most_played_arm));
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
        

        %  Online power allocation
%                       rplot2(K,Gplot, p, index_maxmu, t);
%         errorbar(xx_re.^2+xx_im.^2,S./Nk);
%         hold on
%         errorbar(xx_re.^2+xx_im.^2,sigma2);
%         ylim([0 2]);
%         hold off;
%         pause(0.1)

        % Hinfty-norm estimation
        beta_ts(t) = max(abs(xx_re + 1i*xx_im));
        [~,most_played_arm] = max(Nk);
        beta_ts_arm(t) = abs(xx_re(most_played_arm) + 1i*xx_im(most_played_arm));
    end
    mse_ts = mse_ts + abs(beta_ts - beta).^2;
    mse_ts_arm = mse_ts + abs(beta_ts_arm - beta).^2;
    
    % Regret accumulation
    sR = sR + R;
    sR2 = sR2 + R.^2;
end
mse_ts = mse_ts/Nexp;
mse_ts_arm = mse_ts_arm/Nexp;
toc

%% Power Iterations
%aH = aH/factorH; old command before last minute change
for i = 1:Nexp
uu = rand(L,1);
    for t = 1:T
        y = filter(b,a,uu) + filter(bH,aH,randn(L,1));
        ytilde = flip(y);
        mu = norm(ytilde,2)/sqrt(L);
        uu = ytilde/mu;
        beta_pi(t) = uu'*ytilde/L;
    end
    mse_pi = mse_pi + abs(beta_pi-beta).^2;
end
mse_pi = mse_pi/Nexp;

%% Stephen Tu approach
for i = 1:Nexp
    LL = 10;
    Phi = zeros(L, LL);
    uu = randn(L,1);
    y = filter(b,a,uu) + filter(bH,aH,randn(L,1));
    for ii = LL+1:L
        Phi(ii,:) = uu(ii-LL:ii-1)';
    end
    Phi = Phi(LL+1:end,:);

    theta = Phi\y(LL+1:end);
    theta = flip(theta);
    Ghat = tf(theta',[1,zeros(1,LL-1)],1);
    beta_tu = norm(Ghat,'inf');
    mse_tu = mse_tu + (beta_tu - beta)^2;
end
mse_tu = mse_tu/Nexp;

%% Stephen Tu approach longer filter
for i = 1:Nexp
    LL = 40;
    Phi = zeros(L, LL);
    uu = randn(L,1);
    y = filter(b,a,uu) + filter(bH,aH,randn(L,1));
    for ii = LL+1:L
        Phi(ii,:) = uu(ii-LL:ii-1)';
    end
    Phi = Phi(LL+1:end,:);

    theta = Phi\y(LL+1:end);
    theta = flip(theta);
    Ghat = tf(theta',[1,zeros(1,LL-1)],1);
    beta_tu = norm(Ghat,'inf');
    mse_tu2 = mse_tu2 + (beta_tu - beta)^2;
end
mse_tu2 = mse_tu2/Nexp;

%% SAVE
save('results.mat');

sR = sR/Nexp;
% varR = (sR2 - Nexp*sR.^2)/Nexp;


% figure
% % plot(sR);
% hold on

% time = 1:T;
mult_constant = 0;
for k = 1:K
    if k ~= index_maxmu
        mult_constant = mult_constant + (maxmu^2-abs(G(k))^2) / (   log(     1 + (abs(G(index_maxmu))-abs(G(k)))^2/sigma2(k)    )   );
    end
end
% plot(time,mult_constant*log(time),'g')

% plot(sR+varR);
% plot(sR-varR);

% figure
% plot(downsample(1:T,factor),downsample(sR,factor),'k','linewidth',2)
% hold on
% plot(downsample(1:T,factor),mult_constant*log(downsample(1:T,factor)),'--k','linewidth',2);





%%
figure
plot(downsample(1:T,factor),downsample(sR,factor),'k','linewidth',2)
hold on
plot(downsample(1:T,factor), mult_constant*log(downsample(1:T,factor)),'--k','linewidth',2);
legend('TS', 'Lower bound');
set(gca, 'XScale', 'log')
ylabel('Regret');
xlabel('T')
xlim([0 T]);
legend('TS', '\Delta(log(1+\Delta/\sigma^2))^{-1}');
set(gca, 'XScale', 'log')

figure
freq = linspace(0, (K+1)/length(Gplot)*(length(Gplot)-1), length(Gplot));% %linspace(0, K-1/length(Gplot), length(Gplot));
plot(freq,abs(Gplot),'linewidth',2);
hold on;
armmean = sqrt(xx_re.^2+xx_im.^2);
armvar = S./Nk;
% val = [35 76 115 154 190];
val = ceil(linspace(1,K,5));
errorbar(val, armmean(val), armvar(val),'o','MarkerSize',5,'MarkerEdgeColor','red','MarkerFaceColor','red','linewidth',2);
legend('G(e^{2\pi j k/N})','x','Location',[0.2 0.8 0.1 0.1]);
ylim([0 2.1]);
xlim([0 K+1]);
title('Frequency response of $G$');
text(K-20,0.9,num2str(t))
plot(freq,Hplot,'linewidth',2);


%%
figure
plot(mse_ts);
hold on;
plot(mse_ts_arm);
plot(mse_pi);
plot([1 T], [mse_tu mse_tu]);
plot([1 T], [mse_tu2 mse_tu2]);
legend('TS','TS arm','PI','Tu10','Tu40');
