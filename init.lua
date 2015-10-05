require 'nn'

local TensorTrain, parent = torch.class('nn.TensorTrain', 'nn.Module')

function TensorTrain:__init()
	parent.__init(self)

	self.weight = torch.Tensor()
	self.bias = torch.Tensor()
	self.gradWeight = torch.Tensor()
	self.gradBias = torch.Tensor()

	self.outHeight = nil
	self.outWidth = nil
	self.outChannels = nil

	--TODO: should the constructor arguments resemble Linear or SpatialConvolution?
	--TODO all operations regarding W, tt_transpose, tt_multiply, tt_rank, core2cell
	--TODO: vector indexing
	--TODO: what's currout.dzdx, does it have anything to do with in.dzdx
	--TODO: self:reset()
end

function TensorTrain:updateOutput(input)
	assert(input:dim() == 4)

	local W_core = self.weight
	local inHeight, inWidth, inChannels, batchSize = input:size(1), input:size(2), input:size(3), input:size(4)


	self.output = tt_multiply(W_core, input:view(-1, batchSize))
	if self.bias:nElement() > 0 then
		self.output:add(self.bias:view(self.outHeight, self.outWidth, self.outChannels, 1):expandAs(self.output))
	end
	self.output = self.output:view(self.outHeight, self.outWidth, self.outChannels, batchSize)
	return self.output
end

function TensorTrain:updateGradInput(input, gradOutput)
	local W_core = self.weight
	local inHeight, inWidth, inChannels, batchSize = input:size(1), input:size(2), input:size(3), input:size(4)

	self.gradInput = tt_multiply(tt_transpose(W_core), self.gradInput:view(-1, batchSize))
	self.gradInput = self.gradInput:view(inHeight, inWidth, inChannels, batchSize)
	return self.gradInput
end

function TensorTrain:addGradParameters(input, gradOutput, scale)
	local inHeight, inWidth, inChannels, batchSize = input:size(1), input:size(2), input:size(3), input:size(4)
	if self.bias:nElement() > 0 then
		self.gradBias = self.gradInput:sum(4)
	else
		self.gradBias = []
	end
	
	local DZDWCore = input.new(W_core:size()):zero()
	local rankArr = tt_rank(W)
	local corePos = W.ps

	local numDims = #W.n
	local coreArr = core2cell(W)

	local rightSum = input:view(-1, batchSize)
	rightSum = rightSum:t()

	local leftSum
	for derDim = numDims, 1, -1 do
		if derDim < numDims then
			local rightDim = derDim + 1
			local sumSize = W.m[rightDim] * rankArr[rightDim+1]
			local core = coreArr[rightDim]:view(-1, sumSize)
			rightSum = rightSum:view(-1, W.m[rightDim])
			rightSum = core * (rightSum:t():view(sumSize, -1))
		end

		if derDim >= 2 then
			local core = coreArr[derDim-1]:permute(1, 2, 4, 3)
			core = core:view(-1, W.m[derDim-1])

			leftSum = rightSum:view(rankArr[derDim+1]*torch.prod(W.n(derDim+1:end))*batchSize*torch.prod(W.m:narrow(1, 1, (derDim-2) - (1) + 1)), torch.prod(W.m:narrow(1, derDim-1, (derDim) - (derDim-1) + 1)))
	        	leftSum = core * leftSum:t():view(W.m[derDim-1], -1)

			local leftSumDims = torch.LongStorage{rankArr[derDim-1]*W.n[derDim-1], rankArr[derDim]*W.m[derDim]*rankArr[derDim+1], torch.prod(W.n(derDim+1:end))*batchSize, torch.prod(W.m:narrow(1, 1, (derDim-2) -(1) + 1))}
	        	leftSum = leftSum:view(leftSumDims)
		    	leftSum = leftSum:permute(1, 3, 2, 4)

			for leftDim = derDim-2:1,-1 do
				local sumSize = W.m[leftDim] * rankArr[leftDim+1]
				core = coreArr[leftDim]:view(-1, sumSize)
				leftSum = leftSum:view(-1, W.m[leftDim])
				leftSum = core * leftSum:t():view(sumSize, -1)
			end
		elseif derDim == 1 then
			leftSum = rightSum:view(rankArr[derDim+1], -1, batchSize, W.m[derDim])
	        	leftSum = leftSum:permute(2, 3, 4, 1)
		else
			error('Something bad happened(')
		end
		
		local coreSize = rankArr[derDim] * W.n[derDim] * W.m[derDim] * rankArr[derDim+1]
	    	local leftISize = torch.prod(W.n:narrow(1, 1, (derDim-1) -(1) + 1))
		local rightISize = torch.prod(W.n(derDim+1:end))

		local currout_dzdx = self.gradInput:view(leftISize, W.n[derDim], rightISize*batchSize)

		currout_dzdx = currout_dzdx:permute(2, 1, 3)
		local sumSize = leftISize * rightISize * batchSize
		local der = currout_dzdx:view(-1, sumSize) * leftSum:view(sumSize, -1)

		der = der:view(W.n[derDim], rankArr[derDim], W.m[derDim]*rankArr[derDim+1])
		der = der:permute(2, 1, 3)
		DZDWCore:narrow(1, corePos[derDim], (corePos[derDim+1]-1) - (corePos[derDim]) + 1) = der

		self.gradWeight = DZDWCore
	end
end
