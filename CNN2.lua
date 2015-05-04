require 'nn'
require 'image'
require 'lfs'

-- ACTIVATION FUNCTION
ReLU = nn.ReLU

-- NETWORK TOPOLOGY
-- SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padding])
-- SpatialMaxPooling(kW, kH [, dW, dH])
local model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 3, 240, 240))
model:add(nn.SpatialConvolution(3, 3, 11, 11, 4, 4)):add(ReLU(true))
model:add(nn.SpatialConvolution(3, 48, 5, 5)):add(ReLU(true))
model:add(nn.SpatialMaxPooling(5, 5, 3, 3))
model:add(nn.SpatialConvolution(48, 256, 3, 3)):add(ReLU(true))
model:add(nn.SpatialConvolution(256, 192, 3, 3)):add(ReLU(true))
model:add(nn.SpatialConvolution(192, 192, 3, 3)):add(ReLU(true))
model:add(nn.Linear(4096,4096)):add(nn.ReLU(true))
model:add(nn.Linear(4096,2)):add(nn.ReLU(true))

conv_nodes = model:findModules('nn.SpatialConvolution')
for i = 1, #conv_nodes do
  print(conv_nodes[i].output:size())
end

-- TRAINING THE NETWORK --
for file in lfs.dir(lfs.currentdir().."/FinalData") do
	if (file ~= ".") and (file ~= "..") then 
		local input = image.load("FinalData/"..file, 3); --torch.DoubleTensor
		
		local output= torch.Tensor(2);
		local c = file:sub(1,1)
		local outpustStorage = output:storage()
		if c == "A" then
			outpustStorage[1]=0;
			outpustStorage[2]=1;
		elseif c == "G" then
			outpustStorage[1]=1;
			outpustStorage[2]=0;
		else
			error("Invalid Input Image Filename")
		end
		
		criterion = nn.MSECriterion();
		criterion:forward(model:forward(input), output);
		model:zeroGradParameters();
		model:backward(input, criterion:backward(model.output, output));
		model:updateParameters(0.01);
	end
end


