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
	--TODO: t() vs t_noconj()
	--TODO: separate gradWeight computation from gradInput copmutation
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
	
	local DZDWCore = input.new(W_core:size()):zero()
	rankArr = tt_rank(W)
	corePos = W.ps

	numDims = #W.n
	coreArr = core2cell(W)

	rightSum = input:view(-1, batchSize)
	rightSum = rightSum:t()

	for derDim = numDims, 1, -1 do
		if derDim < numDims then
			rightDim = derDim + 1
			sumSize = W.m[rightDim] * rankArr[rightDim+1]
			core = coreArr[rightDim]:view(-1, sumSize)
			rightSum = rightSum:view(-1, W.m[rightDim])
			rightSum = core * (rightSum:t():view(sumSize, -1))
		end

		if derDim >= 2 then
			core = coreArr[derDim-1]:permute(1, 2, 4, 3)
			core = core:view(-1, W.m[derDim-1])

			leftSum = rightSum:view(rankArr[derDim+1]*torch.prod(W.n(derDim+1:end))*batchSize*torch.prod(W.m(1:derDim-2)), torch.prod(W.m(derDim-1:derDim)))
	        leftSum = core * leftSum:t_noconj():view(W.m[derDim-1], -1)

			leftSumDims = torch.LongStorage{rankArr[derDim-1]*W.n[derDim-1], rankArr[derDim]*W.m[derDim]*rankArr[derDim+1], torch.prod(W.n(derDim+1:end))*batchSize, torch.prod(W.m(1:derDim-2))}
	        leftSum = leftSum:view(leftSumDims)
		    leftSum = leftSum:permute(1, 3, 2, 4)

			for leftDim = derDim-2:1,-1 do
				sumSize = W.m[leftDim] * rankArr[leftDim+1]
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
		
		coreSize = rankArr[derDim] * W.n[derDim] * W.m[derDim] * rankArr[derDim+1]
	    leftISize = torch.prod(W.n(1:derDim-1))
		rightISize = torch.prod(W.n(derDim+1:end))

		currout.dzdx = self.gradInput:view(leftISize, W.n[derDim], rightISize*batchSize)

		currout.dzdx = currout.dzdx:permute(2, 1, 3)
		sumSize = leftISize * rightISize * batchSize
		der = currout.dzdx:view(-1, sumSize) * leftSum:view(sumSize, -1)

		der = der:view(W.n[derDim], rankArr[derDim], W.m[derDim]*rankArr[derDim+1])
		der = der:permute(2, 1, 3)
		DZDWCore(corePos[derDim]:corePos[derDim+1]-1) = der

		self.gradWeight = DZDWCore
	end
end

function TensorTrain:addGradParameters(input, gradOutput, scale)
	if self.bias:nElement() > 0 then
		self.gradBias = self.gradInput:sum(4)
	else
		self.gradBias = []
	end
end
