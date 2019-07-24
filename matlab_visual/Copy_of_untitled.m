clear all
close all
clc

alpha=0.01;
[X, Y] = meshgrid(-2.2:.1:2.2,-2.5:.1:2.5);
syms x y
f = 3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) - 10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) - 1/3*exp(-(x+1).^2 - y.^2);
grad_f = gradient(f);
f_real = subs(f, {x, y}, {X, Y});
Z = double(f_real);
mesh(X,Y,Z)
hold on;
%colormap(gray);
meshgrid off
x0 = zeros(1000,2);
%x0(1,:) = randint(1,2,10);
x0(1,:) = [1 -0.5];
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
sol = [0.2283, -1.6255];
text(sol(1),sol(2),double(subs(f, {x, y}, {sol(1), sol(2)})),'p*','VerticalAlignment','top','HorizontalAlignment','left', 'FontSize', 18)
