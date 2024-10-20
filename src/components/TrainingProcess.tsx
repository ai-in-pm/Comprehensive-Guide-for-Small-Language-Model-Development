import React from 'react';
import { Cpu } from 'lucide-react';

const TrainingProcess: React.FC<{ isActive: boolean }> = ({ isActive }) => {
  if (!isActive) return null;

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-gray-800 flex items-center">
        <Cpu className="mr-2" />
        Training Process
      </h2>
      <p className="text-gray-600">
        Training a Small Language Model (SLM) like CompactLM requires careful consideration of hyperparameters, optimization techniques, and resource management. This section covers the key aspects of the training process.
      </p>
      
      <h3 className="text-2xl font-semibold text-gray-700">1. Configure hyperparameters</h3>
      <p className="text-gray-600">
        Let's set up the hyperparameters for training CompactLM:
      </p>
      <div className="mt-4 p-4 bg-gray-100 rounded-lg">
        <pre className="text-sm overflow-x-auto">
          <code>{`
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    learning_rate=5e-5,
)
          `}</code>
        </pre>
      </div>
      
      <h3 className="text-2xl font-semibold text-gray-700">2. Implement training loop</h3>
      <p className="text-gray-600">
        We'll use the Hugging Face Trainer for our training loop:
      </p>
      <div className="mt-4 p-4 bg-gray-100 rounded-lg">
        <pre className="text-sm overflow-x-auto">
          <code>{`
from transformers import Trainer, DataCollatorForLanguageModeling

# Initialize the model
model = CompactLM()

# Create a data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
          `}</code>
        </pre>
      </div>
      
      <h3 className="text-2xl font-semibold text-gray-700">3. Monitor and visualize training progress</h3>
      <p className="text-gray-600">
        Use TensorBoard to monitor training progress:
      </p>
      <div className="mt-4 p-4 bg-gray-100 rounded-lg">
        <pre className="text-sm overflow-x-auto">
          <code>{`
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def log_metrics(metrics, step):
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)

# In your training loop:
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        loss = train_step(model, batch)
        if step % log_every == 0:
            metrics = {"train/loss": loss}
            log_metrics(metrics, epoch * len(train_dataloader) + step)

writer.close()
          `}</code>
        </pre>
      </div>
      
      <h3 className="text-2xl font-semibold text-gray-700">4. Implement early stopping and learning rate scheduling</h3>
      <p className="text-gray-600">
        Implement early stopping and learning rate scheduling to improve training:
      </p>
      <div className="mt-4 p-4 bg-gray-100 rounded-lg">
        <pre className="text-sm overflow-x-auto">
          <code>{`
from transformers import get_linear_schedule_with_warmup

# Learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_dataloader) * num_epochs
)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping(patience=3, min_delta=0.01)

# In your training loop:
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
    val_loss = evaluate(model, val_dataloader)
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break
          `}</code>
        </pre>
      </div>
      
      <h3 className="text-2xl font-semibold text-gray-700">5. Use distributed training for faster processing</h3>
      <p className="text-gray-600">
        Implement distributed training using PyTorch's DistributedDataParallel:
      </p>
      <div className="mt-4 p-4 bg-gray-100 rounded-lg">
        <pre className="text-sm overflow-x-auto">
          <code>{`
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    model = CompactLM().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Training loop here
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
          `}</code>
        </pre>
      </div>
      
      <h3 className="text-2xl font-semibold text-gray-700">6. Implement checkpointing and model serialization</h3>
      <p className="text-gray-600">
        Regularly save model checkpoints and implement efficient serialization:
      </p>
      <div className="mt-4 p-4 bg-gray-100 rounded-lg">
        <pre className="text-sm overflow-x-auto">
          <code>{`
import os

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

# In your training loop:
for epoch in range(num_epochs):
    # ... training code ...
    if epoch % save_every == 0:
        save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir)
          `}</code>
        </pre>
      </div>
      
      <h3 className="text-2xl font-semibold text-gray-700">7. Develop strategies for handling limited computational resources</h3>
      <p className="text-gray-600">
        Implement techniques like gradient accumulation and mixed-precision training:
      </p>
      <div className="mt-4 p-4 bg-gray-100 rounded-lg">
        <pre className="text-sm overflow-x-auto">
          <code>{`
from torch.cuda.amp import autocast, GradScaler

# Gradient accumulation steps
gradient_accumulation_steps = 4

# Initialize the GradScaler for mixed precision training
scaler = GradScaler()

# In your training loop:
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        with autocast():
            loss = model(batch)
        
        # Normalize the loss to account for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Scales loss and calls backward() to create scaled gradients
        scaler.scale(loss).backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            # Unscales gradients and calls or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
            optimizer.zero_grad()
          `}</code>
        </pre>
      </div>
      
      <div className="mt-8 p-6 bg-green-50 rounded-lg border border-green-200">
        <h4 className="text-xl font-semibold text-green-800 mb-4">Best Practices for Training SLMs</h4>
        <ul className="list-disc list-inside text-green-700 space-y-2">
          <li>Use a learning rate warmup to stabilize early training</li>
          <li>Implement gradient clipping to prevent exploding gradients</li>
          <li>Use weight decay for regularization</li>
          <li>Monitor validation performance closely to prevent overfitting</li>
          <li>Experiment with different optimizers (e.g., AdamW, RAdam) and learning rate schedules</li>
          <li>Use mixed-precision training to speed up computations and reduce memory usage</li>
          <li>Implement data parallelism for multi-GPU training if resources allow</li>
          <li>Use efficient data loading techniques to minimize I/O bottlenecks</li>
          <li>Implement proper logging and visualization for monitoring training progress</li>
          <li>Regularly evaluate on a held-out test set to ensure generalization</li>
        </ul>
      </div>
    </div>
  );
};

export default TrainingProcess;