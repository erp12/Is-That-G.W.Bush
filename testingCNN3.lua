require 'nn'
require 'image'
require 'lfs'

model = torch.load("model.net")
model:evaluate()
print('fileName   prediction   expectation')
for file in lfs.dir(lfs.currentdir().."/FinalData") do
	if (file ~= ".") and (file ~= "..") and  math.random() < 0.05 then 
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
		
		local prediction = model:forward(input)
		print(file)
		print(prediction)
		print(output)
		print("=======================================================")
		
	end
end
