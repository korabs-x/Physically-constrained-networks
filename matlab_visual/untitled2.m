clear all
% Define a colormap that uses the cool colormap and 
% the gray colormap and assign it as the Figure's colormap.
colormap([parula(64);gray(64)])
% Generate some surface data.
[X,Y,Z] = peaks();
% Produce the two surface plots.
h(1) = surf(X,Y,Z);
hold on
h(2) = pcolor(X,Y,Z);
hold off
% Move the pcolor to Z = -10.
% The 0*Z is in the statement below to insure that the size
% of the ZData does not change.
set(h(2),'ZData',-2 + 0*Z)
set(h(2),'FaceColor','interp','EdgeColor','interp')
view(3)
% Scale the CData (Color Data) of each plot so that the 
% plots have contiguous, nonoverlapping values.  The range 
% of each CData should be equal. Here the CDatas are mapped 
% to integer values so that they are easier to manage; 
% however, this is not necessary.
% Initially, both CDatas are equal to Z.
m = 64;  % 64-elements is each colormap
cmin = min(Z(:));
cmax = max(Z(:));
% CData for surface
C1 = round((m-1)*(Z-cmin)/(cmax-cmin))+1; 
% CData for pcolor
C2 = 64+C1;
% Update the CDatas for each object.
set(h(1),'CData',C1);
set(h(2),'CData',C2);
% Change the CLim property of axes so that it spans the 
% CDatas of both objects.
caxis([min(C1(:)) max(C2(:))])