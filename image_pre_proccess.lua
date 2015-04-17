require 'image'
require 'lfs'

for file in lfs.dir(lfs.currentdir()) do
    if lfs.attributes(file,"mode") == "file" and not (file == "image_pre_proccess.lua") then 
		print("Scaling this: "..file)
		rawImg = image.load(file, 3)
		scaledImg = image.scale(rawImg, 240, 240)
		image.save(file, scaledImg)
    end
end
