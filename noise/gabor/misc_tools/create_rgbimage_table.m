function [imout, margout] = create_rgbimage_table(Mim, Nim, Mt, Nt, marg)

% Create a large white rgb image to place MtxNt rgb images of size MimxNim
% with a margin marg (optional argument).

if( nargin == 5)
    margout = marg;
else
    margout = ceil(0.05*max(Mim,Nim));
end

Mout = Mt*Mim + (Mt-1)*margout;
Nout = Nt*Nim + (Nt-1)*margout;

imout = uint8(255*ones(Mout, Nout,3));


end

