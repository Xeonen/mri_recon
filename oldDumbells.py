
class dumbellIMG(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.conv01 = convBlock(16, 3)
        self.sConv01 = convBlock(16, 3, 2)
        self.conv02 = convBlock(32, 3)
        self.sConv02 = convBlock(32, 3, 2)
        self.conv03 = convBlock(64, 3)
        self.sConv03 = convBlock(128, 3, 2)
        
        self.res01 = resBlock(128, 3)
        self.res02 = resBlock(128, 3)
        self.res03 = resBlock(128, 3)
        
        self.tConv01 = convTransBlock(64, 3, 2)
        self.tConv02 = convTransBlock(32, 3, 2)
        self.tConv03 = convTransBlock(16, 3, 2)
        self.conv04 = convBlock(16, 3)
        self.conv05 = convBlock(16, 3)
    
        self.synt = layers.Conv2D(1, 3, padding="same", use_bias=False)
       
    
    def call(self, inputs):
        
        conv01 = self.conv01(inputs)
        sConv01 = self.sConv01(conv01)
        conv02 = self.conv02(sConv01)
        sConv02 = self.sConv02(conv02)
        conv03 = self.conv03(sConv02)
        sConv03 = self.sConv03(conv03)

        
        res01 = self.res01(sConv03)      
        res02 = self.res02(res01)
        res03 = self.res03(res02)  

        

        tConv01 = self.tConv01(res03)
        add01 = layers.Add()([tConv01, conv03])
        tConv02 = self.tConv02(add01)
        add02 = layers.Add()([tConv02, conv02])
        tConv03 = self.tConv03(add02)
        conv04 = self.conv04(tConv03)
        conv05 = self.conv05(conv04)
        synt = self.synt(conv05)

        return(synt)
        
        
  
    
class dumbellS(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.conv01 = convBlock(16, 3)
        self.conv02 = convBlock(16, 3)
        self.sConv01 = convBlock(32, 3, 2)
        
        self.conv03 = convBlock(32, 3)
        self.conv04 = convBlock(32, 3)
        self.sConv02 = convBlock(64, 3, 2)
        
        self.conv05 = convBlock(64, 3)
        self.conv06 = convBlock(64, 3)
        self.sConv03 = convBlock(128, 3, 2)
        
        self.conv07 = convBlock(128, 3)
        self.conv08 = convBlock(128, 3)
        self.sConv04 = convBlock(256, 3, 2)


        
        self.res01 = resBlock(256, 3)
        self.res02 = resBlock(256, 3)
        self.res03 = resBlock(256, 3)
        self.res04 = resBlock(256, 3)
        
        self.tConv01 = convTransBlock(256, 3, 2)
        self.conv09 = convBlock(128, 3)
        self.conv10 = convBlock(128, 3)
        
        
        self.tConv02 = convTransBlock(128, 3, 2)
        self.conv11 = convBlock(64, 3)
        self.conv12 = convBlock(64, 3)
        
        self.tConv03 = convTransBlock(128, 3, 2)
        self.conv13 = convBlock(32, 3)
        self.conv14 = convBlock(32, 3)
        
        self.tConv04 = convTransBlock(32, 3, 2)
        self.conv15 = convBlock(16, 3)
        self.conv16 = convBlock(16, 3)
    
        self.synt = layers.Conv2D(2, 3, padding="same", use_bias=False)
       
    
    def call(self, inputs):
        conv01 = self.conv01(inputs)
        conv02 = self.conv02(conv01)
        sConv01 = self.sConv01(conv02)

        conv03 = self.conv03(sConv01)
        conv04 = self.conv04(conv03)
        sConv02 = self.sConv02(conv04)        
        
        conv05 = self.conv05(sConv02)
        conv06 = self.conv06(conv05)
        sConv03 = self.sConv03(conv06)
        
        conv07 = self.conv07(sConv03)
        conv08 = self.conv08(conv07)
        sConv04 = self.sConv04(conv08)

        
        res01 = self.res01(sConv04)
        res02 = self.res02(res01)
        res03 = self.res03(res02)
        res04 = self.res04(res03)
        
        add01 = layers.Concatenate()([sConv04, res04])
        tConv01 = self.tConv01(add01)
        conv09 = self.conv09(tConv01)
        conv10 = self.conv10(conv09)
        
        
        add02 = layers.Concatenate()([conv08, conv10])
        tConv02 = self.tConv02(add02)
        conv11 = self.conv11(tConv02)
        conv12 = self.conv12(conv11)
        
        add03 = layers.Concatenate()([conv06, conv12])
        tConv03 = self.tConv03(add03)
        conv13 = self.conv13(tConv03)
        conv14 = self.conv14(conv13)
        
        
        add04 = layers.Concatenate()([conv04, conv14])
        tConv04 = self.tConv04(add04)
        conv15 = self.conv15(tConv04)
        conv16 = self.conv16(conv15)
        
        synt = self.synt(conv16)
        

        return(synt)





class dumbellXL(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.conv01 = convBlock(32, 3)
        self.conv02 = convBlock(32, 3)
        self.sConv01 = convBlock(64, 3, 2)
        
        self.conv03 = convBlock(64, 3)
        self.conv04 = convBlock(64, 3)
        self.sConv02 = convBlock(128, 3, 2)
        
        self.conv05 = convBlock(128, 3)
        self.conv06 = convBlock(128, 3)
        self.sConv03 = convBlock(256, 3, 2)
        
        self.conv07 = convBlock(256, 3)
        self.conv08 = convBlock(256, 3)
        self.sConv04 = convBlock(512, 3, 2)


        
        self.res01 = resBlock(512, 3)
        self.res02 = resBlock(512, 3)
        self.res03 = resBlock(512, 3)
        self.res04 = resBlock(512, 3)
        
        self.tConv01 = convTransBlock(512, 3, 2)
        self.conv09 = convBlock(256, 3)
        self.conv10 = convBlock(256, 3)
        
        
        self.tConv02 = convTransBlock(256, 3, 2)
        self.conv11 = convBlock(128, 3)
        self.conv12 = convBlock(128, 3)
        
        self.tConv03 = convTransBlock(128, 3, 2)
        self.conv13 = convBlock(64, 3)
        self.conv14 = convBlock(64, 3)
        
        self.tConv04 = convTransBlock(64, 3, 2)
        self.conv15 = convBlock(32, 3)
        self.conv16 = convBlock(32, 3)
    
        self.synt = layers.Conv2D(2, 3, padding="same", use_bias=False)
       
    
    def call(self, inputs):
        conv01 = self.conv01(inputs)
        conv02 = self.conv02(conv01)
        sConv01 = self.sConv01(conv02)

        conv03 = self.conv03(sConv01)
        conv04 = self.conv04(conv03)
        sConv02 = self.sConv02(conv04)        
        
        conv05 = self.conv05(sConv02)
        conv06 = self.conv06(conv05)
        sConv03 = self.sConv03(conv06)
        
        conv07 = self.conv07(sConv03)
        conv08 = self.conv08(conv07)
        sConv04 = self.sConv04(conv08)

        
        res01 = self.res01(sConv04)
        res02 = self.res02(res01)
        res03 = self.res03(res02)
        res04 = self.res04(res03)
        
        add01 = layers.Concatenate()([sConv04, res04])
        tConv01 = self.tConv01(add01)
        conv09 = self.conv09(tConv01)
        conv10 = self.conv10(conv09)
        
        
        add02 = layers.Concatenate()([conv08, conv10])
        tConv02 = self.tConv02(add02)
        conv11 = self.conv11(tConv02)
        conv12 = self.conv12(conv11)
        
        add03 = layers.Concatenate()([conv06, conv12])
        tConv03 = self.tConv03(add03)
        conv13 = self.conv13(tConv03)
        conv14 = self.conv14(conv13)
        
        
        add04 = layers.Concatenate()([conv04, conv14])
        tConv04 = self.tConv04(add04)
        conv15 = self.conv15(tConv04)
        conv16 = self.conv16(conv15)
        
        synt = self.synt(conv16)
        

        return(synt)
  