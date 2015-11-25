require 'image'
local pl = require('pl.import_into')()

local load_image = function(path, size)
  local size  = size or 224
  local img   = image.load(path)
  local c,w,h = img:size(1), img:size(3), img:size(2)
  assert(c == 3)
  local min   = math.min(w, h)
  img         = image.crop(img, 'c', min, min)
  img         = image.scale(img, size)
  -- normalize (see inception.ipynb -> `ClassifyImageWithInception`)
  img:mul(255):clamp(0, 255):add(-117)
  return img:cuda():view(1, img:size(1), img:size(2), img:size(3))
end

local googlenet = dofile('googlenet.lua')
local net = googlenet({
  cudnn.SpatialConvolution,
  cudnn.SpatialMaxPooling,
  cudnn.ReLU,
  cudnn.SpatialCrossMapLRN
})
net:cuda()

local synsets = pl.utils.readlines('synsets.txt')

-- predict

local scores = net:forward(load_image('dog.jpg'))

scores = scores:float():squeeze()

local _,ind = torch.sort(scores, true)

print('\nRESULTS (top-5):')
print('----------------')
for i=1,5 do
  local synidx = ind[i] + 1 -- synsets is 1-based
  print(string.format(
    "score = %f: %s (%d)", scores[ind[i]], synsets[synidx], ind[i]
  ))
end
