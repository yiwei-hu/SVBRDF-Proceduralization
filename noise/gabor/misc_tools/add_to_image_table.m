function imtableout = add_to_image_table(im, imtable, marg, mt, nt)

% Add the image im to the table imtable with margin marg.
% (mt, nt) are the coordinates in the image table.

% the dimensions are supposed to match.


Mim = size(im,1);
Nim = size(im,2);

mind = (mt-1)*Mim + (mt-1)*marg;
nind = (nt-1)*Nim + (nt-1)*marg;

imtableout = imtable;
imtableout( (mind+1:mind+Mim), (nind+1:nind+Nim) ) = im;


end