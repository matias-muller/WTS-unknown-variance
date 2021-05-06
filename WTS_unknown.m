
clear;
close all;
home;

% Metadata
K = 10; %50;                  % Number of arms
T = 1e5; %1e6;                % K=100, T=1e5, M=70 => ~ 1 minute.    K=100, T=1e6, M=70 => ~10 minutes.
Nexp = 10; %100;               % Number of episodes to average the regret out
sigma2 = 0.05*rand(K,1);     % Noise variances
sigma = sqrt(sigma2);   % Standard deviation of noise
Nsamples = 10; %10;         % Number of samples to approximate rho
tol = 1e-15;            % Minimum allocable power
factor = 100;             % Downsampling plot factor

% 4 secs for K=100, T=1e3, Nsamples=100;

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


% True mean reward distribution
mu_re = real(G);
mu_im = imag(G);
[maxmu, index_maxmu] = max(abs(G));
beta = maxmu;

sR = zeros(T,1);
% sR2 = sR;
n = 3;
p = ones(K,1)/K;

pp = zeros(K,T,Nexp);

tic
for i = 1:Nexp 
    R = zeros(T,1);
    sp = zeros(K,1);
    sx_re = zeros(K,1);
    sx_im = sx_re;
    sS = zeros(K,1);
    S = sS;
    for t = 1:n
        pp(:,t,i) = p; %% saving all p's
        X_re = mu_re +  sqrt(sigma2./(2*p)).*randn(K,1);
        X_im = mu_im +  sqrt(sigma2./(2*p)).*randn(K,1);
        sp = sp + p;
        sx_re = sx_re + p.*X_re;
        sx_im = sx_im + p.*X_im;
        xx_re = sx_re./sp;
        xx_im = sx_im./sp;
        sS = sS + p.*(X_re.^2 + X_im.^2);
        S = sS - sp.*(xx_re.^2 + xx_im.^2);
        R(t) = R(max(t-1,1)) + max(0,maxmu^2 - p'*(abs(G).^2));
    end

        % Udapte p
    samples_re = zeros(K,1);
    samples_im = samples_re;
    r = samples_re;
    u = samples_re;
    rho = samples_re;
    for ii = 1:Nsamples
        % 1. Generate a random radius around xx
        u = rand(K,1);
        r = sqrt(  S./sp .* ((pi*u).^(1/(-t+2)) - 1)  );
        % 2. For a fixed radius, take a uniform sample around the circle
        % parametrizing by theta => (x,y) = (r*cos(theta), r*sin(theta)),
        % with theta = 2*pi*rand();
        theta = 2*pi*rand(K,1);
        samples_re = r.*cos(theta) + xx_re;
        samples_im = r.*sin(theta) + xx_im;
        samples = samples_re.^2 + samples_im.^2;
        [~,ind] = max(samples);
        rho(ind) = rho(ind) + 1;
    end
    p = rho + tol;
    p = p/sum(p);  
    
    
    for t = n+1:T
        pp(:,t,i) = p; %% saving all p's
        
        % Reward generation
        X_re = mu_re +  sqrt(sigma2./(2*p)).*randn(K,1);
        X_im = mu_im +  sqrt(sigma2./(2*p)).*randn(K,1);

        % Update statistics
        sp = sp + p;
        sx_re = sx_re + p.*X_re;
        sx_im = sx_im + p.*X_im;
        xx_re = sx_re./sp;
        xx_im = sx_im./sp;
        xx = sqrt(xx_re.^2 + xx_im.^2);
        sS = sS + p.*(X_re.^2 + X_im.^2);
        S = sS - sp.*(xx_re.^2 + xx_im.^2);
        
        % Expected Cumulative Regret update
        R(t) = R(max(t-1,1)) + max(0,maxmu^2 - p'*(abs(G).^2));

        % Udapte p
        samples_re = zeros(K,1);
        samples_im = samples_re;
        r = samples_re;
        u = samples_re;
        rho = samples_re;
        for ii = 1:Nsamples
            % 1. Generate a random radius around xx
            u = rand(K,1);
            r = sqrt(  S./sp .* ((pi*u).^(1/(-t+2)) - 1)  );
            % 2. For a fixed radius, take a uniform sample around the circle
            % parametrizing by theta => (x,y) = (r*cos(theta), r*sin(theta)),
            % with theta = 2*pi*rand();
            theta = 2*pi*rand(K,1);
            samples_re = r.*cos(theta) + xx_re;
            samples_im = r.*sin(theta) + xx_im;
            samples = samples_re.^2 + samples_im.^2;
            [~,ind] = max(samples);
            rho(ind) = rho(ind) + 1;
        end
        p = rho + tol;
        p = p/sum(p);        

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
%     sR2 = sR2 + R.^2;
end
toc

%% SAVE
save('results.mat');

sR = sR/Nexp;

mult_constant = 0;
for k = 1:K
    if k ~= index_maxmu
        mult_constant = mult_constant + (maxmu-absG(k)) / (   log(     1 + ( maxmu-absG(k) )^2/sigma2(k)    )   );
    end
end

single_constant = 0;
for k = 1:K
    if k ~= index_maxmu
        single_constant = single_constant + sigma2(k) / (  maxmu - absG(k) );
    end
end

figure
plot(downsample(1:T,factor),downsample(sR,factor),'k','linewidth',2)
hold on
plot(downsample(1:T,factor), mult_constant*log(downsample(1:T,factor)),'--k','linewidth',2);
plot(downsample(1:T,factor), single_constant*log(downsample(1:T,factor)),'-.k','linewidth',2);
legend('WTS', '$\log T\sum  \frac{\Delta_k}{\log(1+\Delta_k^2/\sigma_k^2)}$','$\log T \sum \frac{\sigma_k^2}{\Delta}$');
set(legend,'interpreter','latex');
set(gca, 'XScale', 'log');
ylabel('Regret');
xlabel('T')
xlim([0 T]);

% figure
% freq = linspace(0, (K+1)/length(Gplot)*(length(Gplot)-1), length(Gplot));% %linspace(0, K-1/length(Gplot), length(Gplot));
% plot(freq,abs(Gplot),'linewidth',2);
% hold on;
% armmean = sqrt(xx_re.^2+xx_im.^2);
% armvar = zeros(K,1); %S./Nk;
% val = [35 76 115 154 190];
% errorbar(val, armmean(val), armvar(val),'o','MarkerSize',5,'MarkerEdgeColor','red','MarkerFaceColor','red','linewidth',2)
% legend('G(e^{2\pi j k/N})','x','Location',[0.2 0.8 0.1 0.1]);
% ylim([0 1.3]);
% xlim([0 K+1]);
% title('Frequency response of $G$');
% text(K-20,0.9,num2str(t))
