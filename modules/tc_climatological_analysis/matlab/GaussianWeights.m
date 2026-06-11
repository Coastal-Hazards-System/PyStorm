function Wi = GaussianWeights(K_size,distKM)

 Wi = 1/(sqrt(2*pi)*K_size) * exp(-1/2*(distKM/K_size).^2);

end

