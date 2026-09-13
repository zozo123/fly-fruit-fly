import numpy as np
import torch
from fly_fruit_fly.connectome import ConnectomeGraph
from fly_fruit_fly.superfly import SuperFlyPolicy, save_checkpoint, load_checkpoint, set_motor_hidden
from fly_fruit_fly.nonlinear_training import fit_motor


def test_nonlinear_motor_learns_and_round_trips_without_changing_core(tmp_path):
    torch.manual_seed(5)
    graph = ConnectomeGraph(np.arange(4),np.arange(4),np.roll(np.arange(4),1),
                            np.ones(4,dtype=np.float32),{})
    model = SuperFlyPolicy(graph,3,-np.ones(12),np.ones(12))
    set_motor_hidden(model,16)
    before = model.sensory.weight.detach().clone()
    x = torch.cat([torch.randn(128,4),torch.ones(128,1)],1).double()
    y = torch.sin(x[:,:1]).repeat(1,12)*.2
    losses = fit_motor(model,x,y,epochs=40)
    assert losses[-1] < losses[0]
    torch.testing.assert_close(model.sensory.weight,before)
    path = tmp_path/'student.pt'
    save_checkpoint(model,graph,path,{})
    restored = load_checkpoint(path,graph)
    obs = np.zeros(3,dtype=np.float32)
    np.testing.assert_array_equal(model.predict(obs)[0],restored.predict(obs)[0])
