function plotRaster()
% to plot the dynamics of the SNN exported from C++:
system('PDSTEP_demo.exe')
fID = fopen('firings.txt','r');
firings = fscanf(fID,'%f',[2 Inf]);
fclose(fID);
copyfile('firings.txt','old_firings.txt')
delete('firings.txt')

figure
plot(firings(1,:), firings(2,:),'k.')