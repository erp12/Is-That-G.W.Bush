require 'nn'
require 'image'
require 'lfs'

model = torch.load("model.net")											-- Load the network from 'model.net'
model:evaluate()														-- Change the network to testing mode.
print('fileName   prediction   expectation')
for file in lfs.dir(lfs.currentdir().."/FinalData") do					-- For each input image
	if (file ~= ".") and (file ~= "..") and  math.random() < 0.5 then 
		local input = image.load("FinalData/"..file, 3);				-- Load the image
		
		local output= torch.Tensor(2);									-- Generate expected output
		local c = file:sub(1,1)											--		(See documentation in CNN3.lua)
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
		
		local prediction = model:forward(input)							-- Run the input image through the network. 
		print(file)														-- Print input file name
		print(prediction)												-- Print the network's prediction
		print(output)													-- Print what the output should ideally be.
		print("=========================================================")
		
	end
end
