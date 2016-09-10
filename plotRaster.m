function plotRaster()
% to plot the dynamics of the SNN exported from C++.
%
% generate random weights:
weights = rand(100);
dlmwrite('weights.txt',weights,'delimiter',' ');
system('PDSTEP_demo.exe')
fID = fopen('firings.txt','r');
firings = fscanf(fID,'%f',[2 Inf]);
fclose(fID);
copyfile('firings.txt','old_firings.txt')
delete('firings.txt')



% EEG:
EEGpos = zeros(1,max(firings(1,:)));
EEGneg = zeros(1,max(firings(1,:)));
N = max(firings(2,:)); % number of all neurons
for i=1:length(firings(1,:))
    if firings(2,i) < 0.8*N
        EEGpos(firings(1,i)) = EEGpos(firings(1,i)) + 1;
    else
        EEGneg(firings(1,i)) = EEGneg(firings(1,i)) + 1;
    end
end

figure('units','normalized','outerposition',[0 0 1 1])

subplot(3,1,1)
plot(firings(1,:), firings(2,:),'k.')
title('Spike raster plot')
xlabel('Time, ms');
ylabel('Neuron number');

subplot(3,1,2)
plot(EEGpos-EEGneg)
axis([1 max(firings(1,:)) -10 50])
title('Pseudo-EEG signal')
xlabel('Time, ms');
ylabel('Mean spike count');

% Spectrum:
PS = pwelch(EEGpos-EEGneg); 

subplot(3,1,3)
plot(PS(10:end))
axis([0 100 0 20])
title('Power spectrum, linear scale')
xlabel('Frequency (Hz)');
ylabel('Magnitude');
