clear; clc;

%% (a) Symbolic definition
syms x xp k0 real

% Define g1 for x > xp
g1 = -(1i/(2*k0)) * exp(-1i*k0*(x - xp));

% Define g2 for x < xp
g2 = -(1i/(2*k0)) * exp(-1i*k0*(xp - x));

disp('g1 (x > xp) = ');
disp(g1);

disp('g2 (x < xp) = ');
disp(g2);

%% (b) Symbolic verification of Helmholtz equation

% First derivative
dg1 = diff(g1, x);
dg2 = diff(g2, x);

disp('dg1/dx = ');
disp(dg1);

disp('dg2/dx = ');
disp(dg2);

% Second derivative
d2g1 = diff(dg1, x);
d2g2 = diff(dg2, x);

disp('d2g1/dx2 = ');
disp(d2g1);

disp('d2g2/dx2 = ');
disp(d2g2);

% Helmholtz operator
helm1 = simplify(d2g1 + k0^2*g1);
helm2 = simplify(d2g2 + k0^2*g2);

disp('Helmholtz operator on g1:');
disp(helm1);

disp('Helmholtz operator on g2:');
disp(helm2);

%% (c) Derivative discontinuity at x = xp

% Evaluate derivatives at xp 
dg1_at_xp = simplify(subs(dg1, x, xp));
dg2_at_xp = simplify(subs(dg2, x, xp));

disp('dg1/dx at x = xp (right limit):');
disp(dg1_at_xp);

disp('dg2/dx at x = xp (left limit):');
disp(dg2_at_xp);

% Jump
jump = simplify(dg1_at_xp - dg2_at_xp);

disp('Derivative jump (should equal -1):');
disp(jump);

%% (d) Numerical plotting

k0_val = 2*pi;
xp_val = 0;

g_numeric = @(x) -(1i/(2*k0_val)) * exp(-1i*k0_val*abs(x - xp_val));

x_vals = linspace(-3,3,1000);
g_vals = g_numeric(x_vals);

figure;

subplot(2,1,1);
plot(x_vals, real(g_vals));
xlabel('x');
ylabel('Re\{g(x,0)\}');
title('Real Part of Green''s Function');

subplot(2,1,2);
plot(x_vals, imag(g_vals));
xlabel('x');
ylabel('Im\{g(x,0)\}');
title('Imaginary Part of Green''s Function');

%% Symmetry check

g_1_0 = g_numeric(1);
g_0_1 = -(1i/(2*k0_val)) * exp(-1i*k0_val*abs(0 - 1));

disp('g(1,0) = ');
disp(g_1_0);

disp('g(0,1) = ');
disp(g_0_1);

disp('Difference:');
disp(g_1_0 - g_0_1);