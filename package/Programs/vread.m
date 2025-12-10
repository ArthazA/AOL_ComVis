function A = vread(filename);
%
% VREAD: A = vread(filename);
%
% Reads view-format files into matlab.
%
% Author:    M.C. Burl
% Copyright: Jet Propulsion Laboratory, California Institute of Technology
% Date:      1992--1999
%
% You may freely use and redistribute this program.
%
%=====================================================================

% Read parameter file to find number of rows, columns, and precision
pfile = sprintf('%s.spr', filename);
idp = fopen(pfile);
ndim = fscanf(idp, '%d', 1);
if (ndim ~= 2)
  error('Can only read two dimensional data');
end
nc   = fscanf(idp, '%d', 1);  
junk = fscanf(idp, '%f', 1);
junk = fscanf(idp, '%f', 1);
nr   = fscanf(idp, '%d', 1);
junk = fscanf(idp, '%f', 1);
junk = fscanf(idp, '%f', 1);
type = fscanf(idp, '%d', 1);
fclose(idp);

if (type == 0)
  precision = 'unsigned char';
elseif (type == 2)
  precision = 'int';
elseif (type == 3) 
  precision = 'float';
elseif (type == 5)
  precision = 'double';
else 
  error('Unrecognized data type');
end

% Read binary data file
dfile = sprintf('%s.sdt', filename);
idd = fopen(dfile);
A = (fread(idd, [nc, nr], precision))';
fclose(idd);
