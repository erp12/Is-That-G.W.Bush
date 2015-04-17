require 'nn'
require 'image'
require 'lfs'


-- ACTIVATION FUNCTION
ReLU = nn.ReLU

-- NETWORK TOPOLOGY
-- This is what a layer looks like -> nn.VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH [, dT, dW, dH])
-- -- nInputPlane: The number of expected input planes in the image given into forward().
-- -- nOutputPlane: The number of output planes the convolution layer will produce.
-- -- kT: The kernel size of the convolution in time
-- -- kW: The kernel width of the convolution
-- -- kH: The kernel height of the convolution
-- -- dT: The step of the convolution in the time dimension. Default is 1.
-- -- dW: The step of the convolution in the width dimension. Default is 1.
-- -- dH: The step of the convolution in the height dimension. Default is 1.
-- This is what a pooling layer looks like -> nn.VolumetricMaxPooling(kT, kW, kH [, dT, dW, dH])
-- -- kT: The kernel size of the convolution in time
-- -- kW: The kernel width of the convolution
-- -- kH: The kernel height of the convolution
-- -- dT: The step of the convolution in the time dimension. Default is 1.
-- -- dW: The step of the convolution in the width dimension. Default is 1.
-- -- dH: The step of the convolution in the height dimension. Default is 1.
local model = nn.Sequential()
model:add(nn.VolumetricConvolution(1, 96, 11, 11, 3, 4, 4, 4)):add(ReLU(true))
model:add(nn.VolumetricMaxPooling(10, 10, 2))
model:add(nn.VolumetricConvolution(96, 256, 5, 5, 48)):add(ReLU(true))
model:add(nn.VolumetricMaxPooling(4, 4, 47))
model:add(nn.VolumetricConvolution(256, 384, 3, 3, 256)):add(ReLU(true))
model:add(nn.VolumetricConvolution(384, 384, 3, 3, 192)):add(ReLU(true))
model:add(nn.VolumetricConvolution(384, 256, 3, 3, 192)):add(ReLU(true))
model:add(nn.Linear(4096,4096)):add(nn.ReLU(true))
model:add(nn.Linear(4096,2)):add(nn.ReLU(true))

-- TRAINING THE NETWORK --

dataset={};
dataCount = 1;

inputs = {};
outputs = {};

for file in lfs.dir(lfs.currentdir().."/FinalData") do
	if (file ~= ".") and (file ~= "..") then 
		local input = image.load("FinalData/"..file, 3);
		local output= torch.Tensor(2);
		local c = file:sub(1,1)
		if c == "A" then
			output[1]=0;
			output[2]=1;
		elseif c == "G" then
			output[1]=1;
			output[2]=0;
		else
			error("Invalid Input Image Filename")
		end
		dataset[dataCount] = {input, output};
		dataCount = dataCount + 1;
		
		criterion = nn.MSECriterion();
		-- feed it to the neural network and the criterion
		criterion:forward(model:forward(input), output);
		-- (1) zero the accumulation of the gradients
		model:zeroGradParameters();
		-- (2) accumulate gradients
		model:backward(input, criterion:backward(model.output, output));
		-- (3) update parameters with a 0.01 learning rate
		model:updateParameters(0.01);
	end
end

-- An alternate way to train, which I couldn't get to work.
--[[
function dataset:size() return dataCount end

criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)
]]--

