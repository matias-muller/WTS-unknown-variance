
% pp is a set of Nexp matrices, each of which is K x T storting p^t_k.
% Initialized: pp = zeros(K,T,Nexp);
%
% meanpp is the K x T sample mean of these matrices along i = 1:Nexp
%
% varpp is the K x T sample variance of each entry along i = 1:Nexp, or 
% E[ (p^t_k - meanpp(k,t))^2 ]
% 
% secpp is the sample second moment of each entry along i = 1:Nexp, or
% E[ (p^t_k)^2 ].
%
% We aim to plot f(t) = (1/t)Sum_{ell=1}^t Sqrt[ E[ (p^t_k)^2 ] ]

meanpp = sum(pp,3)/Nexp;
varpp = sum((pp-meanpp).^2,3)/Nexp;
secpp = sum(pp.^2,3)/Nexp;
sq_sum = (cumsum(pp,2)).^2;
avg_sq_sum = sum(sq_sum,3)/Nexp;

figure
for k = 1:K
    if(k ~= index_maxmu)
% %       Plotting (1/t)*E[  (sum^t p^n_k)^2   ]
         plot(1:T, avg_sq_sum(k,:)./(1:T));
%         semilogy(1:T, avg_sq_sum(k,:)./(1:T));
%         
% %         Plotting upper bound (1/t)*Sum^t Sqrt( E[(p^t_k)^2] )
%         semilogy( 1:T,  cumsum(sqrt(secpp(k,:)))./(1:T) );
%         plot( 1:T,  cumsum(sqrt(secpp(k,:)))./(1:T) );
        hold on
    end
end