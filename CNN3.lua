require 'nn'
require 'image'
require 'lfs'

-- ACTIVATION FUNCTION
ReLU = nn.ReLU

-- NETWORK TOPOLOGY
-- nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padding])
-- nn.SpatialMaxPooling(kW, kH [, dW, dH])
local model = nn.Sequential()
model:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
model:add(ReLU(true))
model:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
model:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
model:add(ReLU(true))
model:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
model:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
model:add(ReLU(true))
model:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
model:add(ReLU(true))
model:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
model:add(ReLU(true))
model:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

model:add(nn.View(256*6*6))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(256*6*6, 4096))
model:add(nn.Threshold(0, 1e-6))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 4096))
model:add(nn.Threshold(0, 1e-6))
model:add(nn.Linear(4096, 2))
model:add(nn.LogSoftMax())

-- TRAINING THE NETWORK --

testingDataFileNames = {}
shouldTrain = true;


criterion = nn.MSECriterion();

for file in lfs.dir(lfs.currentdir().."/FinalData") do
	if (file ~= ".") and (file ~= "..") and shouldTrain then 
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
		--criterion = nn.MSECriterion();
		criterion:forward(model:forward(input), output);
		model:zeroGradParameters();
		model:backward(input, criterion:backward(model.output, output));
		model:updateParameters(0.01);
	end
end

torch.save("model.net", model)
