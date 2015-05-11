require 'image'
require 'lfs'

for file in lfs.dir(lfs.currentdir()) do
    if lfs.attributes(file,"mode") == "file" and not (file == "image_pre_proccess.lua") then 
		print("Scaling this: "..file)
		rawImg = image.load(file, 3)									-- Load the image.
		scaledImg = image.scale(rawImg, 240, 240)						-- Scale the image to 240px by 240px.
		image.save(file, scaledImg)										-- Save the image with same filname from input.
    end
end
