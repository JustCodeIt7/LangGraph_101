# Subgraph To-Do App with LangGraph
## Combined 
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
                    +-----------+                    
                    | __start__ |                    
                    +-----------+                    
                          *                          
                          *                          
                          *                          
                     +---------+                     
                     | analyze |                     
                     +---------+                     
                    ..         ..                    
                  ..             ..                  
                 .                 .                 
        +-----------+          +--------+            
        | breakdown |          | simple |            
        +-----------+          +--------+            
                    **         **                    
                      **     **                      
                        *   *                        
                   +------------+                    
                   | prioritize |                    
                   +------------+                    
                          *                          
                          *                          
                          *                          
                 +-----------------+                 
                 | display_results |                 
                 +-----------------+                 
                          *                          
                          *                          
                          *                          
                     +---------+                     
                     | __end__ |                     
                     +---------+                     
</pre>
## Using SubGraphs in LangGraph
### enrichment_graph --> enrich_task:
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">           +-----------+           
           | __start__ |           
           +-----------+           
                 *                 
                 *                 
                 *                 
            +---------+            
            | analyze |            
            +---------+            
           ..         ..           
         ..             ..         
        .                 .        
+-----------+          +--------+  
| breakdown |          | simple |  
+-----------+          +--------+  
           **         **           
             **     **             
               *   *               
            +---------+            
            | __end__ |            
            +---------+            
</pre>

### reminder_graph --> generate_reminder:
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
+-----------+  
| __start__ |  
+-----------+  
       *       
       *       
       *       
+------------+ 
| prioritize | 
+------------+ 
       *       
       *       
       *       
  +---------+  
  | __end__ |  
  +---------+  
</pre>

### main_graph:
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    +-----------+      

    | __start__ |      
    +-----------+      
          *            
          *            
          *            
   +-------------+     
   | enrich_task |     
   +-------------+     
          *            
          *            
          *            
+-------------------+  
| generate_reminder |  
+-------------------+  
          *            
          *            
          *            
 +-----------------+   
 | display_results |   
 +-----------------+   
          *            
          *            
          *            
     +---------+       
     | __end__ |       
     +---------+       
</pre>

