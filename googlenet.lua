hdf5 = require 'hdf5'
require 'cudnn'
require 'cunn'

local function loadWeights(model, f)
   local function loadWeight(m)
      if m.gname then
         local w = paths.concat(f, m.gname .. '_w.h5')
         local myFile = hdf5.open(w, 'r')
         w = myFile:read(m.gname .. '_w'):all()
         myFile:close()

         local b = paths.concat(f, m.gname .. '_b.h5')
         local myFile = hdf5.open(b, 'r')
         b = myFile:read(m.gname .. '_b'):all()
         myFile:close()

         -- w and b are now tensors
         w = w:transpose(2,4):transpose(3,4):clone()
         w = w:typeAs(m.weight)
         if m.gname == 'softmax2' then
            w = w:squeeze()
         end

         assert(w:isSameSizeAs(m.weight))
         m.weight:copy(w)

         b = b:typeAs(m.bias)
         assert(b:isSameSizeAs(m.bias))
         m.bias:copy(b)
      end
   end
   model:apply(loadWeight)
end

local function inception(depth_dim, input_size, config, lib, name)
   local SpatialConvolution = lib[1]
   local SpatialMaxPooling = lib[2]
   local ReLU = lib[3]

   local depth_concat = nn.Concat(depth_dim)
   local conv1 = nn.Sequential()
   local m = SpatialConvolution(input_size, config[1][1], 1, 1)
   m.gname = 'mixed' .. name .. '_' .. '1x1'
   conv1:add(m):add(ReLU(true))
   depth_concat:add(conv1)

   local conv3 = nn.Sequential()
   local m = SpatialConvolution(input_size, config[2][1], 1, 1)
   m.gname = 'mixed' .. name .. '_' .. '3x3_bottleneck'
   conv3:add(m):add(ReLU(true))
   local m = SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1)
   m.gname = 'mixed' .. name .. '_' .. '3x3'
   conv3:add(m):add(ReLU(true))
   depth_concat:add(conv3)

   local conv5 = nn.Sequential()
   local m = SpatialConvolution(input_size, config[3][1], 1, 1)
   m.gname = 'mixed' .. name .. '_' .. '5x5_bottleneck'
   conv5:add(m):add(ReLU(true))
   local m = SpatialConvolution(config[3][1], config[3][2], 5, 5, 1, 1, 2, 2)
   m.gname = 'mixed' .. name .. '_' .. '5x5'
   conv5:add(m):add(ReLU(true))
   depth_concat:add(conv5)

   local pool = nn.Sequential()
   pool:add(SpatialMaxPooling(config[4][1], config[4][1], 1, 1, 1, 1))
   local m = SpatialConvolution(input_size, config[4][2], 1, 1)
   m.gname = 'mixed' .. name .. '_' .. 'pool_reduce'
   pool:add(m):add(ReLU(true))
   depth_concat:add(pool)

   return depth_concat
end

local function googlenet(lib)
   local SpatialConvolution = lib[1]
   local SpatialMaxPooling = lib[2]
   local SpatialAveragePooling = torch.type(lib[2]()) == 'nn.SpatialMaxPooling' and nn.SpatialAveragePooling or cudnn.SpatialAveragePooling
   local ReLU = lib[3]
   local SpatialCrossMapLRN = lib[4]
   local model = nn.Sequential()
   local m = SpatialConvolution(3,64,7,7,2,2,3,3)
   m.gname = 'conv2d0'
   model:add(m):add(ReLU(true))
   model:add(SpatialMaxPooling(3,3,2,2,1,1))
   model:add(SpatialCrossMapLRN(11, 0.00109999999404, 0.5, 2.0))
   local m = SpatialConvolution(64,64,1,1,1,1,0,0)
   m.gname = 'conv2d1'
   model:add(m):add(ReLU(true))
   local m = SpatialConvolution(64,192,3,3,1,1,1,1)
   m.gname = 'conv2d2'
   model:add(m):add(ReLU(true))
   model:add(SpatialCrossMapLRN(11, 0.00109999999404, 0.5, 2.0))
   model:add(SpatialMaxPooling(3,3,2,2,1,1))
   model:add(inception(2, 192, {{ 64}, { 96,128}, {16, 32}, {3, 32}},lib, '3a')) -- 256
   model:add(inception(2, 256, {{128}, {128,192}, {32, 96}, {3, 64}},lib, '3b')) -- 480
   model:add(SpatialMaxPooling(3,3,2,2))
   model:add(inception(2, 480, {{192}, { 96,204}, {16, 48}, {3, 64}},lib, '4a')) -- 4(a)
   model:add(inception(2, 508, {{160}, {112,224}, {24, 64}, {3, 64}},lib, '4b')) -- 4(b)
   model:add(inception(2, 512, {{128}, {128,256}, {24, 64}, {3, 64}},lib, '4c')) -- 4(c)
   model:add(inception(2, 512, {{112}, {144,288}, {32, 64}, {3, 64}},lib, '4d')) -- 4(d)
   model:add(inception(2, 528, {{256}, {160,320}, {32,128}, {3,128}},lib, '4e')) -- 4(e) (14x14x832)
   model:add(SpatialMaxPooling(3,3,2,2,1,1))
   model:add(inception(2, 832, {{256}, {160,320}, {48,128}, {3,128}},lib, '5a')) -- 5(a)
   model:add(inception(2, 832, {{384}, {192,384}, {48,128}, {3,128}},lib, '5b')) -- 5(b)
   model:add(SpatialAveragePooling(7,7,1,1))
   model:add(nn.View(1024):setNumInputDims(3))
   -- model:add(nn.Dropout(0.4))
   local m = nn.Linear(1024,1008)
   m.gname = 'softmax2'
   model:add(m)
   model:add(nn.Narrow(2, 2, 1000)) -- because google has 1008 classes, class 2 - 1001 are valid
   model:add(nn.SoftMax())

   loadWeights(model, 'dump/')

   return model,'GoogleNet', {128,3,224,224}
end

return googlenet
