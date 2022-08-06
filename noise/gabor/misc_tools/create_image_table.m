function [imout, margout] = create_image_table(Mim, Nim, Mt, Nt, marg)

% Create a large white image to place MtxNt images of size Mim, Nim with a
% margin marg (optional argument).

if( nargin == 5)
    margout = marg;
else
    margout = ceil(0.05*max(Mim,Nim));
end

Mout = Mt*Mim + (Mt-1)*margout;
Nout = Nt*Nim + (Nt-1)*margout;

imout = 255*ones(Mout, Nout);


end



















