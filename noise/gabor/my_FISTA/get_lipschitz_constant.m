function L = get_lipschitz_constant(A, At, m, n, quiet)

% Return twice the maximal value of the operator At*A using the power
% iteration method.

% Inspired from F. Malgouyres's get_operator_norm.m funciton available at:
% http://www.math.univ-paris13.fr/~malgouy/


nbIterMin = 30;
nbIterMax = 3000;
acc = 1e-10;

% First step:
it = 1;
Lold = 0;
% v = randn(n,1);
v = ones(n,1); % not good in general but better for our Gabor noise problem
v = v/norm(v);
u = A*v;
v = At*u;
L = 2*norm(v);
if quiet ~= 1 ;
        disp( ['Iteration ', num2str(it), ' ; L = ', num2str(L)] ); 
end

% Main loop:
while  ( ((abs(Lold - L) > acc) || (it < nbIterMin)) && (it < nbIterMax) )
    it=it+1;
    v = v/norm(v);
    u = A*v;
    v = At*u;
    Lold = L;
    L = 2*norm(v);
    if quiet ~= 1 ;
        disp( ['Iteration ', num2str(it), ' ; L = ', num2str(L)] ); 
    end
end

if quiet ~= 1 ;
    disp( 'Lipshitz constant computed' ); 
end

end
