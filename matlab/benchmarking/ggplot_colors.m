function cols = ggplot_colors(n, varargin)
% ref: http://stackoverflow.com/questions/16861822/
%      emulate-ggplot2-default-color-palette-in-matlab
%
p = inputParser; 
p.addRequired('n', @isnumeric);
p.addOptional('hue', [0.1 0.9], @(x) length(x) == 2 & min(x) >=0 & max(x) <= 1);
p.addOptional('saturation', 0.5, @(x) length(x) == 1);
p.addOptional('value', 0.8, @(x) length(x) == 1);

p.parse(n, varargin{:});

cols = hsv2rgb( ...
    [transpose(linspace(p.Results.hue(1), p.Results.hue(2), p.Results.n)), ...
     repmat(p.Results.saturation, p.Results.n, 1), ...
     repmat(p.Results.value, n,1)]);