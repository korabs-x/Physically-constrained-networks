clear all
close all
clc

m = 64;  % 64-elements is each colormap
%colormap([parula(m);gray(m)])
colormap(gray(m))

alpha=0.01;
step = 0.1;%.1;
[X, Y] = meshgrid(-2.2:step:2.2,-2.5:step:2.5);
syms x y

constr1 = 0*x;%0.5*x - y - 1.1;
constr2 = (x - y - 1.85);
constr3 = 0*x;%(x + y - 1.6);
constr4 = 0*x;%-(x - 0.9*y + 0.5);

x0 = zeros(1000,2);
%x0(1,:) = randint(1,2,10);
x0(1,:) = [1, -0.5];
x0(1,:) = [-1, 1];

f_orig = 3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) - 10*(x/5 - x.^3 - y.^5).*exp(-(x-0.1).^2-y.^2) - 2/3*exp(-(x+1).^2 - y.^2) + 3*exp(1.8*(- (x-0.2)^2 - (y-0.6)^2)) - exp(1.8*(- (x+1.4)^2 - (y-0.2)^2));
% default
f = f_orig;
% penalty method
%mu = 1;
mu = 3;
%f = f_orig + mu*(piecewise(constr1 > 0, constr1, 0)^2 + piecewise(constr2 > 0, constr2, 0)^2 + piecewise(constr3 > 0, constr3, 0)^2 + piecewise(constr4 > 0, constr4, 0)^2);
% lagrangian relaxation
%f = f_orig + 5.938 * constr4;
% augmented lagrangian
iteration = 5;
mu = 0.125 * 2^iteration;
lambda = 0;
%2nd iteration
if iteration == 1
    x0(1,:) = [-1.2959, 0.1591];
    lambda = 0.4131;
elseif iteration == 2
    x0(1,:) = [-1.2501, 0.0902];
    lambda = 1.2107;
elseif iteration == 3
    x0(1,:) = [-1.1698, -0.0380];
    lambda = 2.7016;
elseif iteration == 4
    x0(1,:) = [-1.0144, -0.4127];
    lambda = 5.1533;
elseif iteration == 5
    x0(1,:) = [0.3810, -1.6704];
    lambda = 4.7506;
end
f = f_orig - 0.5 * lambda * constr2 + mu * constr2 * constr2;

grad_f = gradient(f);

f_real = subs(f, {x, y}, {X, Y});
Z = double(f_real);
% Plots
h(1) = mesh(X,Y,Z);
hold on;

cmin = min(Z(:));
cmax = max(Z(:));
C1 = min(m-1, round((m-1)*(Z-cmin)/(cmax-cmin))+1);
%for i = 1:size(C1,1)
%    for j = 1:size(C1,2)
%        if (double(subs(constr4, {x, y}, {X(i,j), Y(i,j)})) > 0) || (double(subs(constr3, {x, y}, {X(i,j), Y(i,j)})) > 0) || (double(subs(constr2, {x, y}, {X(i,j), Y(i,j)})) > 0)
        %if (double(subs(constr1, {x, y}, {X(i,j), Y(i,j)})) > 0) || (double(subs(constr2, {x, y}, {X(i,j), Y(i,j)})) > 0) || (double(subs(constr3, {x, y}, {X(i,j), Y(i,j)})) > 0) || (double(subs(constr4, {x, y}, {X(i,j), Y(i,j)})) > 0)
        %if (double(subs(constr1, {x, y}, {X(i,j), Y(i,j)})) > 0) || double(subs(constr2, {x, y}, {X(i,j), Y(i,j)})) > 0) || double(subs(constr3, {x, y}, {X(i,j), Y(i,j)})) > 0) || double(subs(constr4, {x, y}, {X(i,j), Y(i,j)})) > 0)
        %if (constr1(X(i,j),Y(i,j)) > 0) || (constr2(X(i,j),Y(i,j)) > 0) || (constr3(X(i,j),Y(i,j)) > 0) || (constr4(X(i,j),Y(i,j)) > 0)
%            C1(i, j) = C1(i, j) + m;
            %C1
%        end
%    end
%end

set(h(1),'CData',C1);

%caxis([min(C1(:)) max(C2(:))])

meshgrid off

xPlane = [3 -1.15 -1.15 3];      % X coordinates of plane corners, ordered around the plane
yPlane = [1.15 -3 -3 1.15];      % Corresponding y coordinates for plane 1
zPlane = [-10 -10 15 15];  % Z coordinates of plane corners
hold on;                   % Add to existing plot
patch(xPlane, yPlane, zPlane, 'g', 'FaceAlpha', 0.3);  % Plot plane 1

plot3(x0(1,1),x0(1,2),double(subs(f, {x, y}, {x0(1,1), x0(1,2)})),'mx','MarkerSize',10);
i=2;
while(i < 100)
    % Gradient descent equation..
    % val = double(subs(f, {x, y}, {x0(i-1,1), x0(i-1,2)}))
    g = double(subs(grad_f, {x, y}, {x0(i-1,1), x0(i-1,2)}));
    x0(i,:) = x0(i-1,:) - alpha.*g.';
    plot3(x0(i,1),x0(i,2),double(subs(f, {x, y}, {x0(i,1), x0(i,2)})),'mx','MarkerSize',10)
    i=i+1;
end
%x0(i-1, :)
lambda_new = lambda - mu * double(subs(constr2, {x, y}, {x0(i-1, 1), x0(i-1, 2)}));
%lambda_new
mu
lambda
double(subs(constr2, {x, y}, {x0(i-1, 1), x0(i-1, 2)}))
sol = [0.2283, -1.6255];
plot3(sol(1),sol(2),double(subs(f, {x, y}, {sol(1), sol(2)})),'mx','MarkerSize',10, 'Color', [255, 153, 51] / 255);
text(sol(1),sol(2),double(subs(f, {x, y}, {sol(1), sol(2)})),'w*','VerticalAlignment','top','HorizontalAlignment','left', 'FontSize', 18, 'Color', [255, 153, 51] / 255)
solc = [0.0409, -1.4591];
%plot3(solc(1),solc(2),double(subs(f, {x, y}, {solc(1), solc(2)})),'mx','MarkerSize',10, 'Color', [255, 153, 51] / 255);
%text(solc(1),solc(2),double(subs(f, {x, y}, {solc(1), solc(2)})),'p^c','VerticalAlignment','top','HorizontalAlignment','right', 'FontSize', 18, 'Color', [255, 153, 51] / 255)
xlim manual
xlim([-3, 3])
ylim manual
ylim([-3, 3])
zlim([-10, 15])

xlabel('w_1')
ylabel('w_2')
zlabel('Loss')
