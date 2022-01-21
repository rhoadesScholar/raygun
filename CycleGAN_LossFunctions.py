# !conda activate n2v
import torch

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.Logger('CycleGAN_Loss', 'INFO')

class CycleGAN_Loss(torch.nn.Module):
    def __init__(self, 
                l1_loss, 
                gan_loss, 
                netD1, 
                netG1, 
                netD2, 
                netG2, 
                optimizer_D1, 
                optimizer_G1, 
                optimizer_D2, 
                optimizer_G2, 
                dims,
                l1_lambda=100, 
                identity_lambda=0,
                padding=None,
                gan_mode=None
                 ):
        super(CycleGAN_Loss, self).__init__()
        self.l1_loss = l1_loss
        self.gan_loss = gan_loss
        self.netD1 = netD1 # differentiates between fake and real Bs
        self.netG1 = netG1 # turns As into Bs
        self.netD2 = netD2 # differentiates between fake and real As
        self.netG2 = netG2 # turns Bs into As
        self.optimizer_D1 = optimizer_D1
        self.optimizer_G1 = optimizer_G1
        self.optimizer_D2 = optimizer_D2
        self.optimizer_G2 = optimizer_G2
        self.l1_lambda = l1_lambda
        self.identity_lambda = identity_lambda
        self.padding = padding
        self.gan_mode = gan_mode
        self.dims = dims
        if (self.padding is not None) and (self.padding.lower() != 'valid'):
            inds = [...]
            for pad in self.padding:
                inds.append(slice(pad, -pad))
            self.pad_inds = tuple(inds)
        self.loss_dict = {
            'Loss/D1': float(),
            'Loss/D2': float(),
            'Loss/cycle': float(),
            'Loss/G1': float(),
            'Loss/G2': float(),
        }

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-self.dims] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def clamp_weights(self, net, min=-0.01, max=0.01):
        for module in net.model:
            if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                temp = module.weight.data
                module.weight.data = temp.clamp(min, max)

    def backward_D(self, Dnet, real, fake, cycled):
        # Real
        pred_real = Dnet(real)
        loss_D_real = self.gan_loss(pred_real, True)
        
        # Fake; stop backprop to the generator by detaching fake
        pred_fake = Dnet(fake.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)

        # Cycled; stop backprop to the generator by detaching cycled
        pred_cycled = Dnet(cycled.detach())
        loss_D_cycled = self.gan_loss(pred_cycled, False)

        loss_D = loss_D_real + loss_D_fake + loss_D_cycled
        loss_D.backward()
        return loss_D

    def backward_Ds(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B, n_loop=5):
        self.set_requires_grad([self.netD1, self.netD2], True)  # enable backprop for D
        self.optimizer_D1.zero_grad()     # set D's gradients to zero
        self.optimizer_D2.zero_grad()     # set D's gradients to zero

        #Do D1 first
        if self.gan_mode.lower() == 'wgangp':
            for _ in range(n_loop):
                loss_D1 = self.backward_D(self.netD1, real_B, fake_B, cycled_B)
                self.optimizer_D1.step()          # update D's weights
                self.clamp_weights(self.netD1)
        else:
            loss_D1 = self.backward_D(self.netD1, real_B, fake_B, cycled_B)
            self.optimizer_D1.step()          # update D's weights
            

        #Then D2
        if self.gan_mode.lower() == 'wgangp':
            for _ in range(n_loop):
                loss_D2 = self.backward_D(self.netD2, real_A, fake_A, cycled_A)
                self.optimizer_D2.step()
                self.clamp_weights(self.netD2)
        else:
            loss_D2 = self.backward_D(self.netD2, real_A, fake_A, cycled_A)
            self.optimizer_D2.step()
        
        #return losses
        return loss_D1, loss_D2

    def backward_G(self, Dnet, fake, cycled, cycle_loss, identity_loss=None):
        """Calculate GAN and L1 loss for the generator"""        
        # First, G(A) should fake the discriminator
        pred_fake = Dnet(fake)
        gan_loss_fake = self.gan_loss(pred_fake, True)

        # Second, G(F(B)) should also fake the discriminator 
        pred_cycled = Dnet(cycled)
        gan_loss_cycle = self.gan_loss(pred_cycled, True)
        
        # Include L1 loss for forward and reverse cycle consistency
        if self.identity_lambda > 0:
            loss_G = cycle_loss + gan_loss_fake + gan_loss_cycle + identity_loss
        else:
            loss_G = cycle_loss + gan_loss_fake + gan_loss_cycle

        # calculate gradients
        loss_G.backward(retain_graph=True)
        return loss_G

    def backward_Gs(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):
        self.set_requires_grad([self.netD1, self.netD2], False)  # D requires no gradients when optimizing G
        self.optimizer_G1.zero_grad()        # set G's gradients to zero
        self.optimizer_G2.zero_grad()        # set G's gradients to zero

        #get cycle loss for both directions (i.e. real == cycled, a.k.a. real_A == netG2(netG1(real_A)) for A and B)
        if self.padding is not None and self.padding.lower() == 'valid':
            l1_loss_A = self.l1_loss(self.crop(real_A, cycled_A.size()[-self.dims:]), cycled_A)
            l1_loss_B = self.l1_loss(self.crop(real_B, cycled_B.size()[-self.dims:]), cycled_B)
        else:
            l1_loss_A = self.l1_loss(real_A, cycled_A)
            l1_loss_B = self.l1_loss(real_B, cycled_B)        
        self.loss_dict.update({
            'Cycle_Loss/A': float(l1_loss_A),                
            'Cycle_Loss/B': float(l1_loss_B),                
        })
        cycle_loss = self.l1_lambda * (l1_loss_A + l1_loss_B)

        #get identity loss (i.e. ||G_A(B) - B|| for G_A(A) --> B) if applicable
        if self.identity_lambda > 0:
            identity_B = self.netG1(real_B)
            identity_A = self.netG2(real_A)
            if self.padding is not None and self.padding.lower() == 'valid':
                identity_loss_B = self.l1_loss(self.crop(real_B, identity_B.size()[-self.dims:]), identity_B)#TODO: add ability to have unique loss function for identity
                identity_loss_A = self.l1_loss(self.crop(real_A, identity_A.size()[-self.dims:]), identity_A)
            else:
                identity_loss_B = self.l1_loss(real_B, identity_B)#TODO: add ability to have unique loss function for identity
                identity_loss_A = self.l1_loss(real_A, identity_A)
            self.loss_dict.update({
                'Identity_Loss/A': float(identity_loss_A),                
                'Identity_Loss/B': float(identity_loss_B),                
            })
        else:
            identity_loss_B = None
            identity_loss_A = None

        #Then G1 first
        loss_G1 = self.backward_G(self.netD1, fake_B, cycled_B, cycle_loss, identity_loss_B)                   # calculate gradient for G

        #Then G2
        loss_G2 = self.backward_G(self.netD2, fake_A, cycled_A, cycle_loss, identity_loss_A)                   # calculate gradient for G
        
        #Step optimizers
        self.optimizer_G1.step()             # udpate G's weights
        self.optimizer_G2.step()             # udpate G's weights

        #return losses
        return cycle_loss, loss_G1, loss_G2

    def forward(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):#s, mask_A, mask_B):

        # real_A_mask = real_A * mask_A
        # cycled_A_mask = cycled_A * mask_A
        # fake_A_mask = fake_A * mask_B # masked based on mask from "real" version of array before generator pass
        # real_B_mask = real_B * mask_B
        # cycled_B_mask = cycled_B * mask_B
        # fake_B_mask = fake_B * mask_A
        if (self.padding is not None) and (self.padding.lower() != 'valid'):
            real_A = real_A[self.pad_inds]
            fake_A = fake_A[self.pad_inds]
            cycled_A = cycled_A[self.pad_inds]
            real_B = real_B[self.pad_inds]
            fake_B = fake_B[self.pad_inds]
            cycled_B = cycled_B[self.pad_inds]

        # # update Ds
        # loss_D1, loss_D2 = self.backward_Ds(real_A_mask, fake_A_mask, cycled_A_mask, real_B_mask, fake_B_mask, cycled_B_mask)
        loss_D1, loss_D2 = self.backward_Ds(real_A, fake_A, cycled_A, real_B, fake_B, cycled_B)

        # update Gs
        # cycle_loss, loss_G1, loss_G2 = self.backward_Gs(real_A_mask, fake_A_mask, cycled_A_mask, real_B_mask, fake_B_mask, cycled_B_mask)
        cycle_loss, loss_G1, loss_G2 = self.backward_Gs(real_A, fake_A, cycled_A, real_B, fake_B, cycled_B)
        
        self.loss_dict.update({
            'Loss/D1': float(loss_D1),
            'Loss/D2': float(loss_D2),
            'Loss/cycle': float(cycle_loss),
            'Loss/G1': float(loss_G1),
            'Loss/G2': float(loss_G2),
        })

        total_loss = cycle_loss + loss_G1 + loss_G2 #+ loss_D1 + loss_D2
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        logger.info(self.loss_dict)
        return total_loss

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

class SplitGAN_Loss(torch.nn.Module):
    def __init__(self, 
                l1_loss, 
                gan_loss, 
                netD1, 
                netG1, 
                netD2, 
                netG2, 
                optimizer_D1, 
                optimizer_G1, 
                optimizer_D2, 
                optimizer_G2, 
                dims,
                l1_lambda=100, 
                identity_lambda=0,
                padding=None,
                gan_mode=None
                 ):
        super(SplitGAN_Loss, self).__init__()
        self.l1_loss = l1_loss
        self.gan_loss = gan_loss
        self.netD1 = netD1 # differentiates between fake and real Bs
        self.netG1 = netG1 # turns As into Bs
        self.netD2 = netD2 # differentiates between fake and real As
        self.netG2 = netG2 # turns Bs into As
        self.optimizer_D1 = optimizer_D1
        self.optimizer_G1 = optimizer_G1
        self.optimizer_D2 = optimizer_D2
        self.optimizer_G2 = optimizer_G2
        self.l1_lambda = l1_lambda
        self.identity_lambda = identity_lambda
        self.padding = padding
        self.gan_mode = gan_mode
        self.dims = dims
        if (self.padding is not None) and (self.padding.lower() != 'valid'):
            inds = [...]
            for pad in self.padding:
                inds.append(slice(pad, -pad))
            self.pad_inds = tuple(inds)
        self.loss_dict = {
            'Loss/D1': float(),
            'Loss/D2': float(),
            'Loss/G1': float(),
            'Loss/G2': float(),
        }

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-self.dims] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]
                
    def clamp_weights(self, net, min=-0.01, max=0.01):
        for module in net.model:
            if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                temp = module.weight.data
                module.weight.data = temp.clamp(min, max)

    def backward_D(self, Dnet, real, fake, cycled):
        # Real
        pred_real = Dnet(real)
        loss_D_real = self.gan_loss(pred_real, True)
        
        # Fake; stop backprop to the generator by detaching fake
        pred_fake = Dnet(fake.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)

        # Cycled; stop backprop to the generator by detaching cycled
        pred_cycled = Dnet(cycled.detach())
        loss_D_cycled = self.gan_loss(pred_cycled, False)

        loss_D = loss_D_real + loss_D_fake + loss_D_cycled
        loss_D.backward()
        return loss_D

    def backward_Ds(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B, n_loop=5):
        self.set_requires_grad([self.netD1, self.netD2], True)  # enable backprop for D
        self.optimizer_D1.zero_grad()     # set D's gradients to zero
        self.optimizer_D2.zero_grad()     # set D's gradients to zero

        #Do D1 first
        if self.gan_mode.lower() == 'wgangp':
            for _ in range(n_loop):
                loss_D1 = self.backward_D(self.netD1, real_B, fake_B, cycled_B)
                self.optimizer_D1.step()          # update D's weights
                self.clamp_weights(self.netD1)
        else:
            loss_D1 = self.backward_D(self.netD1, real_B, fake_B, cycled_B)
            self.optimizer_D1.step()          # update D's weights
            

        #Then D2
        if self.gan_mode.lower() == 'wgangp':
            for _ in range(n_loop):
                loss_D2 = self.backward_D(self.netD2, real_A, fake_A, cycled_A)
                self.optimizer_D2.step()
                self.clamp_weights(self.netD2)
        else:
            loss_D2 = self.backward_D(self.netD2, real_A, fake_A, cycled_A)
            self.optimizer_D2.step()

        #return losses
        return loss_D1, loss_D2

    def backward_G(self, side, Gnet, Dnet, real, fake, cycled):
        """Calculate GAN and L1 loss for the generator"""        
        # First, G(A) should fake the discriminator
        pred_fake = Dnet(fake)
        gan_loss_fake = self.gan_loss(pred_fake, True)

        # Second, G(F(B)) should also fake the discriminator 
        pred_cycled = Dnet(cycled)
        gan_loss_cycle = self.gan_loss(pred_cycled, True)
        
        # Include L1 loss for forward and reverse cycle consistency
        if self.padding is not None and self.padding.lower() == 'valid':
            cycle_loss = self.l1_lambda * self.l1_loss(self.crop(real, cycled.size()[-self.dims:]), cycled)
        else:
            cycle_loss = self.l1_lambda * self.l1_loss(real, cycled)                 
        self.loss_dict.update({
            'Cycle_Loss/'+side: float(cycle_loss)            
        })
        
        # Combine losses
        loss_G = cycle_loss + gan_loss_fake + gan_loss_cycle
        #get identity loss (i.e. ||G_A(B) - B|| for G_A(A) --> B) and add if applicable
        if self.identity_lambda > 0:
            identity = Gnet(real)
            if self.padding is not None and self.padding.lower() == 'valid':
                identity_loss = self.l1_loss(self.crop(real, identity.size()[-self.dims:]), identity)
            else:
                identity_loss = self.l1_loss(real, identity)#TODO: add ability to have unique loss function for identity             
            self.loss_dict.update({
                'Identity_Loss/'+side: float(identity_loss)            
            })
            loss_G = loss_G + identity_loss

        # calculate gradients
        loss_G.backward(retain_graph=True)
        return loss_G

    def backward_Gs(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):
        self.set_requires_grad([self.netD1, self.netD2], False)  # D requires no gradients when optimizing G
        self.optimizer_G1.zero_grad()        # set G's gradients to zero
        self.optimizer_G2.zero_grad()        # set G's gradients to zero

        #G1 first
        loss_G1 = self.backward_G('B', self.netG1, self.netD1, real_B, fake_B, cycled_B)                   # calculate gradient for G

        #Then G2
        loss_G2 = self.backward_G('A', self.netG2, self.netD2, real_A, fake_A, cycled_A)                   # calculate gradient for G

        #Step optimizers
        self.optimizer_G1.step()             # udpate G1's weights
        self.optimizer_G2.step()             # udpate G2's weights

        #return losses
        return loss_G1, loss_G2

    def forward(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):#s, mask_A, mask_B):

        # real_A_mask = real_A * mask_A
        # cycled_A_mask = cycled_A * mask_A
        # fake_A_mask = fake_A * mask_B # masked based on mask from "real" version of array before generator pass
        # real_B_mask = real_B * mask_B
        # cycled_B_mask = cycled_B * mask_B
        # fake_B_mask = fake_B * mask_A
        if (self.padding is not None) and (self.padding.lower() != 'valid'):
            real_A = real_A[self.pad_inds]
            fake_A = fake_A[self.pad_inds]
            cycled_A = cycled_A[self.pad_inds]
            real_B = real_B[self.pad_inds]
            fake_B = fake_B[self.pad_inds]
            cycled_B = cycled_B[self.pad_inds]

        # # update Ds
        # loss_D1, loss_D2 = self.backward_Ds(real_A_mask, fake_A_mask, cycled_A_mask, real_B_mask, fake_B_mask, cycled_B_mask)
        loss_D1, loss_D2 = self.backward_Ds(real_A, fake_A, cycled_A, real_B, fake_B, cycled_B)

        # update Gs
        # cycle_loss, loss_G1, loss_G2 = self.backward_Gs(real_A_mask, fake_A_mask, cycled_A_mask, real_B_mask, fake_B_mask, cycled_B_mask)
        loss_G1, loss_G2 = self.backward_Gs(real_A, fake_A, cycled_A, real_B, fake_B, cycled_B)
        
        self.loss_dict.update({
            'Loss/D1': float(loss_D1),
            'Loss/D2': float(loss_D2),
            'Loss/G1': float(loss_G1),
            'Loss/G2': float(loss_G2),
        })

        total_loss = loss_G1 + loss_G2 #+ loss_D1 + loss_D2
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        logger.info(self.loss_dict)
        return total_loss

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    