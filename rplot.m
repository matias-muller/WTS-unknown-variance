function rplot(K, Gplot, p, index_maxmu, i)
    freq = linspace(0, (K+1)/length(Gplot)*(length(Gplot)-1), length(Gplot));% %linspace(0, K-1/length(Gplot), length(Gplot));
    plot(freq,abs(Gplot));
    hold on;
    stem(p(i,:));
    stem(index_maxmu,p(i,index_maxmu),'k','linewidth',4);
    hold off;
    legend('G(e^{2\pi j k/N})','p','best','Location',[0.2 0.8 0.1 0.1]);
    ylim([0 1]);
    xlim([0 K+1]);
    title('PI power allocation');
    text(K-20,0.9,num2str(i))
    hold off;
    pause(0.01);    
end