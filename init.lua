require 'nn'

local TensorTrain, parent = torch.class('nn.TensorTrain', 'nn.Module')

function TensorTrain:__init(loader, nThreads)
	parent.__init(self)
end

function TensorTrain:updateOutput(input)
	return self.output
end

