require 'nn'
require 'image'
require 'lfs'

-- ACTIVATION FUNCTION
ReLU = nn.ReLU -- f(x)= max(0,x)

-- NETWORK TOPOLOGY
-- nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padding])
-- nn.SpatialMaxPooling(kW, kH [, dW, dH])
local model = nn.Sequential()
model:add(nn.SpatialConvolution(3,96,11,11,4,4,2,2))	
model:add(ReLU(true))
-- ^^^ create 96 kernals of 11 by 11. Activate with ReLU.
model:add(nn.SpatialMaxPooling(3,3,2,2))
-- ^^^ cut resolution in half.
model:add(nn.SpatialConvolution(96,192,5,5,1,1,2,2))
model:add(ReLU(true))
-- ^^^ create 192 kernals of 5 by 5. Activate with ReLU.
model:add(nn.SpatialMaxPooling(3,3,2,2))
-- ^^^ cut resolution in half.
model:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))	
model:add(ReLU(true))
-- ^^^ create 384 kernals of 3 by 3. Activate with ReLU.
model:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))  	--	
model:add(ReLU(true))
-- ^^^ create 256 kernals of 3 by 3. Activate with ReLU
model:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))  	--	
model:add(ReLU(true))
-- ^^^ create 256 kernals of 3 by 3. Activate with ReLU
model:add(nn.SpatialMaxPooling(3,3,2,2))               	--	
-- ^^^ cut resolution in half.

-- CLASSIFIER LAYERS --
-- The follow layers describe 3 fully connected, feed forward, layers.
model:add(nn.View(256*6*6))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(256*6*6, 4096)) -- Layer 1
model:add(nn.Threshold(0, 1e-6))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 4096))	-- Layer 2
model:add(nn.Threshold(0, 1e-6))
model:add(nn.Linear(4096, 2))		-- Layer 3
model:add(nn.LogSoftMax())

-- TRAINING THE NETWORK --

--criterion = nn.MSECriterion();

for file in lfs.dir(lfs.currentdir().."/FinalData") do					-- For every file in FinalData Folder
	if (file ~= ".") and (file ~= "..") and math.random() < 0.8 then 							-- Exclude current directory and parent directory
		local input = image.load("FinalData/"..file, 3);				-- Use image package to load into tensor (multi-dimentional data structure)

		local output= torch.Tensor(2);									-- expected output will be 1 by 2 tensor.
		local c = file:sub(1,1)											-- Get first character of input file name.
		local outpustStorage = output:storage()							-- outputStorage is the actual values of
		if c == "A" then
			outpustStorage[1]=0;										-- If input image filename starts with an 'A' (not 'G' for George W. Bush)
			outpustStorage[2]=1;										--		expected output should be [0, 1] for 0% GWB, 100% not GWB.
		elseif c == "G" then
			outpustStorage[1]=1;										-- If input image filename stats with a 'G' for George W. Bush
			outpustStorage[2]=0;										--		expectd outut should be [1, 0] for 100% GWB, 0% not GWB.
		else
			error("Invalid Input Image Filename")						-- If filname starts with something other than 'A' or 'G', that file should not be present.
		end
		criterion = nn.MSECriterion();
		criterion:forward(model:forward(input), output);				-- Push input through the network, and save how far off from output the result is. 
		model:zeroGradParameters();
		model:backward(input, criterion:backward(model.output, output));-- Back-propagate if each node should be higher or lower.
		model:updateParameters(0.01);									-- Tweak weights with learning-rate 0.01.
	end
end

torch.save("model.net", model)											-- Save the weights in file 'model.net' to be tested later.
